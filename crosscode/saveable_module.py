"""
A lightweight wrapper around torch.nn.Module that allows for more robust and easier saving and loading of models. Built
around the observation that a state_dict is not enough to fully describe a model.

Saves a model under a directory, in two files:
- model.pt: the model parameters
- model_cfg.yaml: the model configuration

The model.pt is the standard state_dict, whereas the model_cfg.yaml contains all the information needed to
instantiate a blank instance of a model which will be compatible with that state_dict.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore
from torch import nn

if sys.version_info.minor < 11:
    Self = Any
else:
    from typing import Self


class SaveableModule(nn.Module, ABC):
    STATE_DICT_FNAME = "model.pt"
    MODEL_CFG_FNAME = "model_cfg.yaml"

    @abstractmethod
    def _dump_cfg(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def _scaffold_from_cfg(cls: type[Self], cfg: dict[str, Any]) -> Self: ...

    def save(self, basepath: Path):
        basepath.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), basepath / self.STATE_DICT_FNAME)
        with open(basepath / self.MODEL_CFG_FNAME, "w") as f:
            yaml.dump(self._dump_cfg(), f)

    @classmethod
    def load(cls: type[Self], basepath: Path | str, device: torch.device | str = "cpu") -> Self:
        basepath = Path(basepath)
        with open(basepath / cls.MODEL_CFG_FNAME) as f:
            cfg = yaml.safe_load(f)
        model = cls._scaffold_from_cfg(cfg)
        model.load_state_dict(torch.load(basepath / cls.STATE_DICT_FNAME, weights_only=True, map_location=device))
        return model

    def clone(self) -> Self:
        out = self._scaffold_from_cfg(self._dump_cfg())
        out.load_state_dict(self.state_dict())
        return out


# Add a custom constructor for the !!python/tuple tag,
# converting the loaded sequence to a Python tuple.
def _tuple_constructor(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/tuple", _tuple_constructor)
