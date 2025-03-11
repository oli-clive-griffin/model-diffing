from typing import Any

import torch as t
from torch import nn

from model_diffing.models.activations.activation_function import ActivationFunction


def rectangle(x: t.Tensor) -> t.Tensor:
    """
    when:
        x < -0.5 -> 0
        -0.5 < x < 0.5 -> 1
        x > 0.5 -> 0
    """
    return ((x > -0.5) & (x < 0.5)).to(x)


class AnthropicJumpReLUActivation(ActivationFunction):
    def __init__(
        self,
        size: int,
        bandwidth: float,
        log_threshold_init: float | None = None,
        backprop_through_input: bool = True,
    ):
        super().__init__()

        self.size = size
        self.bandwidth = bandwidth
        self.log_threshold_H = nn.Parameter(t.ones(size))
        if log_threshold_init is not None:
            with t.no_grad():
                self.log_threshold_H.mul_(log_threshold_init)
        self.backprop_through_input = backprop_through_input

    def forward(self, hidden_preact_BH: t.Tensor) -> t.Tensor:
        return AnthropicJumpReLU.apply(
            hidden_preact_BH, self.log_threshold_H, self.bandwidth, self.backprop_through_input
        )  # type: ignore

    def _dump_cfg(self) -> dict[str, int | float | str | list[float]]:
        return {
            "size": self.size,
            "bandwidth": self.bandwidth,
            "backprop_through_input": self.backprop_through_input,
        }

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "AnthropicJumpReLUActivation":
        return cls(
            size=cfg["size"],
            bandwidth=cfg["bandwidth"],
            log_threshold_init=None,  # will be handled by loading the state dict
            backprop_through_input=cfg["backprop_through_input"],
        )


class AnthropicJumpReLU(t.autograd.Function):
    """
    NOTE: this implementation does not support directly optimizing L0 as in the original GDM "Jumping Ahead" paper.
    To do this, we'd need to return the step and the output, then take step.sum(dim=-1). Taking L0 of output will not
    work as L0 is not differentiable. If you want to optimize L0 directly, you need to define a seperate Autograd
    function that returns `(input_BX > threshold_X)` and implements the STE backward pass for that.
    """

    @staticmethod
    def forward(
        ctx: Any,
        input_BX: t.Tensor,
        log_threshold_X: t.Tensor,
        bandwidth: float,
        backprop_through_input: bool,
    ) -> t.Tensor:
        """
        threshold_X is $\\theta$ in the GDM paper, $t$ in the Anthropic paper.

        Where GDM don't backprop through the threshold in "jumping ahead", Anthropic do in the jan 2025 update.
        """
        threshold_X = log_threshold_X.exp()
        ctx.save_for_backward(input_BX, threshold_X, t.tensor(bandwidth))
        ctx.backprop_through_input = backprop_through_input
        return (input_BX > threshold_X) * input_BX

    @staticmethod
    def backward(ctx: Any, grad_output_BX: t.Tensor) -> tuple[t.Tensor | None, t.Tensor, None, None]:  # type: ignore
        input_BX, threshold_X, bandwidth = ctx.saved_tensors

        grad_threshold_BX = (
            threshold_X  #
            * -(1 / bandwidth)
            * rectangle((input_BX - threshold_X) / bandwidth)
            * grad_output_BX
        )

        grad_threshold_X = grad_threshold_BX.sum(0)  # this is technically unnecessary as torch will automatically do it

        if ctx.backprop_through_input:
            return (
                (input_BX > threshold_X) * grad_output_BX,  # input_BX
                grad_threshold_X,
                None,  # bandwidth
                None,  # backprop_through_input
            )

        return (
            None,  # input_BX
            grad_threshold_X,
            None,  # bandwidth
            None,  # backprop_through_input
        )
