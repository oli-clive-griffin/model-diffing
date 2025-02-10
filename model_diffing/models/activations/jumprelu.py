from typing import Any

import torch as t
from torch import nn

from model_diffing.utils import SaveableModule


def rectangle(x: t.Tensor) -> t.Tensor:
    """
    when:
        x < -0.5 -> 0
        -0.5 < x < 0.5 -> 1
        x > 0.5 -> 0
    """
    return ((x > -0.5) & (x < 0.5)).to(x)


class JumpReLUActivation(SaveableModule):
    def __init__(
        self,
        size: int,
        bandwidth: float,
        threshold_init: float,
        backprop_through_input: bool = True,
    ):
        super().__init__()
        self.log_threshold_H = nn.Parameter(t.ones(size) * threshold_init)
        self.bandwidth = bandwidth
        self.backprop_through_input = backprop_through_input

    def forward(self, x_BX: t.Tensor) -> t.Tensor:
        return JumpReLU.apply(x_BX, self.log_threshold_H, self.bandwidth, self.backprop_through_input)  # type: ignore

    def _dump_cfg(self) -> dict[str, int | float | str]:
        return {"size": self.log_threshold_H.shape[0], "bandwidth": self.bandwidth}

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "JumpReLUActivation":
        return cls(**cfg)


class JumpReLU(t.autograd.Function):
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
