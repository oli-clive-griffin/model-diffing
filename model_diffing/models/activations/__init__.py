from .activation_function import ActivationFunction
from .jumprelu import AnthropicJumpReLUActivation
from .relu import ReLUActivation
from .topk import BatchTopkActivation, GroupMaxActivation, TopkActivation

_classes: list[type[ActivationFunction]] = [
    AnthropicJumpReLUActivation,
    BatchTopkActivation,
    GroupMaxActivation,
    ReLUActivation,
    TopkActivation,
]

ACTIVATIONS_MAP: dict[str, type[ActivationFunction]] = {
    **{cls.__name__: cls for cls in _classes},
    "JumpReLUActivationFunction": AnthropicJumpReLUActivation,
}
