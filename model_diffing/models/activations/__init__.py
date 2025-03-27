from typing import cast

from .activation_function import ActivationFunction
from .jumprelu import AnthropicSTEJumpReLUActivation
from .relu import ReLUActivation
from .topk import BatchTopkActivation, GroupMaxActivation, TopkActivation

_classes: list[type[ActivationFunction]] = [
    cast(type[ActivationFunction], AnthropicSTEJumpReLUActivation),
    cast(type[ActivationFunction], BatchTopkActivation),
    cast(type[ActivationFunction], GroupMaxActivation),
    cast(type[ActivationFunction], ReLUActivation),
    cast(type[ActivationFunction], TopkActivation),
]

ACTIVATIONS_MAP: dict[str, type[ActivationFunction]] = {
    **{cls.__name__: cls for cls in _classes},
    "JumpReLUActivationFunction": cast(type[ActivationFunction], AnthropicSTEJumpReLUActivation),
}
