from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

import torch
import yaml
from pydantic import BaseModel
from torch import nn

from model_diffing.log import logger


def save_model_and_config(config: BaseModel, save_dir: Path, model: nn.Module, epoch: int) -> None:
    """Save the model to disk. Also save the config file if it doesn't exist.

    Args:
        config: The config object. Saved if save_dir / "config.yaml" doesn't already exist.
        save_dir: The directory to save the model and config to.
        model: The model to save.
        epoch: The current epoch (used in the model filename).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    if not (save_dir / "config.yaml").exists():
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)
        logger.info("Saved config to %s", save_dir / "config.yaml")

    model_file = save_dir / f"model_epoch_{epoch + 1}.pt"
    torch.save(model.state_dict(), model_file)
    logger.info("Saved model to %s", model_file)


T = TypeVar("T")


def chunk(iterable: Iterator[T], size: int) -> Iterator[list[T]]:
    chunk: list[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def l1_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=1, dim=dim, keepdim=keepdim, out=out, dtype=dtype)


def l2_norm(
    input: torch.Tensor,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    out: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.norm(input, p=2, dim=dim, keepdim=keepdim, out=out, dtype=dtype)
