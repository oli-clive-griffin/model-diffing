from typing import Any

import torch
from torch import nn

from crosscoding.models.activations.activation_function import ActivationFunction


def rectangle(x: torch.Tensor) -> torch.Tensor:
    """
    when:
        x < -0.5 -> 0
        -0.5 < x < 0.5 -> 1
        x > 0.5 -> 0
    """
    return ((x > -0.5) & (x < 0.5)).to(x)


class AnthropicSTEJumpReLUActivation(ActivationFunction):
    def __init__(
        self,
        size: int,
        bandwidth: float,
        log_threshold_init: float | None = None,
    ):
        super().__init__()

        self.size = size
        self.bandwidth = bandwidth
        self.log_threshold_L = nn.Parameter(torch.ones(size))
        if log_threshold_init is not None:
            with torch.no_grad():
                self.log_threshold_L.mul_(log_threshold_init)

    def forward(self, latent_preact_BL: torch.Tensor) -> torch.Tensor:
        return AnthropicJumpReLU.apply(latent_preact_BL, self.log_threshold_L, self.bandwidth)  # type: ignore

    def _dump_cfg(self) -> dict[str, int | float | str | list[float]]:
        return {
            "size": self.size,
            "bandwidth": self.bandwidth,
        }

    @classmethod
    def _scaffold_from_cfg(cls, cfg: dict[str, Any]) -> "AnthropicSTEJumpReLUActivation":
        return cls(
            size=cfg["size"],
            bandwidth=cfg["bandwidth"],
            log_threshold_init=None,  # will be handled by loading the state dict
        )


class AnthropicJumpReLU(torch.autograd.Function):
    """
    NOTE: this implementation does not support directly optimizing L0 as in the original GDM "Jumping Ahead" paper.
    To do this, we'd need to return the step and the output, then take step.sum(dim=-1). Taking L0 of output will not
    work as L0 is not differentiable. If you want to optimize L0 directly, you need to define a seperate Autograd
    function that returns `(input_BX > threshold_X)` and implements the STE backward pass for thatorch.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input_BX: torch.Tensor,
        log_threshold_X: torch.Tensor,
        bandwidth: float,
    ) -> torch.Tensor:
        """
        threshold_X is $\\theta$ in the GDM paper, $t$ in the Anthropic paper.

        Where GDM don't backprop through the threshold in "jumping ahead", Anthropic do in the jan 2025 update.
        """
        threshold_X = log_threshold_X.exp()
        ctx.save_for_backward(input_BX, threshold_X, torch.tensor(bandwidth))
        return (input_BX > threshold_X) * input_BX

    @staticmethod
    def backward(ctx: Any, output_grad_BX: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor, None]:  # type: ignore
        input_BX, threshold_X, bandwidth = ctx.saved_tensors

        threshold_local_grad_BX = -(threshold_X / bandwidth) * rectangle((input_BX - threshold_X) / bandwidth)
        input_local_grad_BX = input_BX > threshold_X

        input_grad_BX = input_local_grad_BX * output_grad_BX
        threshold_grad_BX = threshold_local_grad_BX * output_grad_BX

        return (
            input_grad_BX,
            threshold_grad_BX,
            None,  # bandwidth
        )
