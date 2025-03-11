from typing import cast

from .activation_function import ActivationFunction
from .jumprelu import AnthropicJumpReLUActivation
from .relu import ReLUActivation
from .topk import BatchTopkActivation, GroupMaxActivation, TopkActivation

_classes: list[type[ActivationFunction]] = [
    cast(type[ActivationFunction], AnthropicJumpReLUActivation),
    cast(type[ActivationFunction], BatchTopkActivation),
    cast(type[ActivationFunction], GroupMaxActivation),
    cast(type[ActivationFunction], ReLUActivation),
    cast(type[ActivationFunction], TopkActivation),
]

ACTIVATIONS_MAP: dict[str, type[ActivationFunction]] = {
    **{cls.__name__: cls for cls in _classes},
    "JumpReLUActivationFunction": AnthropicJumpReLUActivation,
}
