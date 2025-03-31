from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass
class CrosscodingDim:
    name: str
    index_labels: list[str]

    def __len__(self) -> int:
        return len(self.index_labels)

class CrosscodingDims(OrderedDict[str, CrosscodingDim]):
    def index(self, dim_name: str) -> int:
        return list(self.keys()).index(dim_name)

class BaseActivationsDataloader(ABC):
    @abstractmethod
    def get_activations_iterator_BXD(self) -> Iterator[torch.Tensor]: ...

    @abstractmethod
    def get_norm_scaling_factors_X(self) -> torch.Tensor: ...

    @abstractmethod
    def get_crosscoding_dims_X(self) -> CrosscodingDims: ...
