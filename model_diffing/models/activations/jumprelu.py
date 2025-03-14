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
    ):
        super().__init__()

        self.size = size
        self.bandwidth = bandwidth
        self.log_threshold_H = nn.Parameter(t.ones(size))
        if log_threshold_init is not None:
            with t.no_grad():
                self.log_threshold_H.mul_(log_threshold_init)

    def forward(self, hidden_preact_BH: t.Tensor) -> t.Tensor:
        return AnthropicJumpReLU.apply(hidden_preact_BH, self.log_threshold_H, self.bandwidth)  # type: ignore

    def _dump_cfg(self) -> dict[str, int | float | str | list[float]]:
        return {
            "size": self.size,
            "bandwidth": self.bandwidth,
        }

    @classmethod
    def _from_cfg(cls, cfg: dict[str, Any]) -> "AnthropicJumpReLUActivation":
        return cls(
            size=cfg["size"],
            bandwidth=cfg["bandwidth"],
            log_threshold_init=None,  # will be handled by loading the state dict
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
    ) -> t.Tensor:
        """
        threshold_X is $\\theta$ in the GDM paper, $t$ in the Anthropic paper.

        Where GDM don't backprop through the threshold in "jumping ahead", Anthropic do in the jan 2025 update.
        """
        threshold_X = log_threshold_X.exp()
        ctx.save_for_backward(input_BX, threshold_X, t.tensor(bandwidth))
        return (input_BX > threshold_X) * input_BX

    @staticmethod
    def backward(ctx: Any, output_grad_BX: t.Tensor) -> tuple[t.Tensor | None, t.Tensor, None]:  # type: ignore
        input_BX, threshold_X, bandwidth = ctx.saved_tensors

        threshold_local_grad_BX = -(threshold_X / bandwidth) * rectangle((input_BX - threshold_X) / bandwidth)
        input_local_grad_BX = (input_BX > threshold_X)


        input_grad_BX = input_local_grad_BX * output_grad_BX
        threshold_grad_BX = threshold_local_grad_BX * output_grad_BX

        # print(f'{input_local_grad_BX=}')
        # print(f'{input_grad_BX=}')

        return (
            input_grad_BX,
            threshold_grad_BX,
            None,  # bandwidth
        )


# # sanity check STE
# def new_func():
#     import math
#     jr = AnthropicJumpReLUActivation(
#         size=1,
#         bandwidth=2.0,
#         log_threshold_init=math.log(1),
#     )

#     optim = t.optim.SGD(jr.parameters(), lr=0.01)

#     for _ in range(1000):
#         optim.zero_grad()

#         x = t.tensor([[0.5]], requires_grad=True)
#         out = jr.forward(x)
#         loss = (0.5 - out) ** 2
#         loss.backward()
#         optim.step()

#     if jr.log_threshold_H.exp().item() > 0.5:
#         raise ValueError("t > 0.5")
#     return jr

# def new_func1():
#     threshold = 1.0
#     # optimize inputs to be above threshold
#     x = nn.Parameter(t.tensor([[0.9]]))
#     y_hat = t.tensor([[1.1]])


#     jr = AnthropicJumpReLUActivation(
#         size=1,
#         bandwidth=2.0,
#         log_threshold_init=math.log(threshold),
#     )

#     optim = t.optim.SGD([x], lr=0.01)

#     for _ in range(1000):
#         optim.zero_grad()
#         y = jr.forward(x)
#         loss = (y_hat - y) ** 2
#         loss.backward()
#         print(f"d(x)={x.grad.item():.3f}, x={x.item():.3f}, y={y.item():.3f}")
#         optim.step()
#         sleep(0.01)
    
#     if (x - y_hat).abs() < 1e-6:
#         raise ValueError("didnt optimize x correctly")

# # sanity check STE
# def new_func2():
#     jr = AnthropicJumpReLUActivation(
#         size=1,
#         bandwidth=2.0,
#         log_threshold_init=math.log(0.01),
#     )

#     optim = t.optim.SGD(jr.parameters(), lr=0.01)

#     for i in range(100000):
#         optim.zero_grad()

#         x = t.tensor([[0.5]], requires_grad=True)
#         out = jr.forward(x)
#         loss = out ** 2
#         loss.backward()
#         optim.step()
#         if i % 1000 == 0:
#             print(f"t={jr.log_threshold_H.exp().item():.3f}, x={x.item():.3f}, out={out.item():.3f}")
#         # print(jr.log_threshold_H.exp().item())
#         # sleep(0.01)



# if __name__ == "__main__":
#     from time import sleep

#     new_func2()
