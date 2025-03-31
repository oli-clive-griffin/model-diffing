from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CrosscodingDim:
    name: str
    index_labels: list[str]

    def __len__(self) -> int:
        return len(self.index_labels)

class CrosscodingDimsDict(OrderedDict[str, CrosscodingDim]):
    def index(self, dim_name: str) -> int:
        return list(self.keys()).index(dim_name)

    def sizes(self) -> tuple[int, ...]:
        return tuple(len(dim) for dim in self.values())

    @classmethod
    def from_dims(cls, *dims: CrosscodingDim) -> "CrosscodingDimsDict":
        return cls(OrderedDict((dim.name, dim) for dim in dims))

