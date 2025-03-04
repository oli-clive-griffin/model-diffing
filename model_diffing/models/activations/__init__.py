from model_diffing.utils import SaveableModule

from .jumprelu import AnthropicJumpReLUActivation
from .relu import ReLUActivation
from .topk import BatchTopkActivation, TopkActivation

classes: list[type[SaveableModule]] = [
    TopkActivation,
    BatchTopkActivation,
    ReLUActivation,
    AnthropicJumpReLUActivation,
]

ACTIVATIONS_MAP: dict[str, type[SaveableModule]] = {cls.__name__: cls for cls in classes}
