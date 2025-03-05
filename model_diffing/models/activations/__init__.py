from .activation_function import ActivationFunction
from .jumprelu import AnthropicJumpReLUActivation
from .relu import ReLUActivation
from .topk import BatchTopkActivation, TopkActivation

_classes: list[type[ActivationFunction]] = [
    TopkActivation,
    BatchTopkActivation,
    ReLUActivation,
    AnthropicJumpReLUActivation,
]

ACTIVATIONS_MAP: dict[str, type[ActivationFunction]] = {cls.__name__: cls for cls in _classes}
