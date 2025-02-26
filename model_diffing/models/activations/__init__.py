from model_diffing.utils import SaveableModule

from .jumprelu import JumpReLUActivation
from .relu import ReLUActivation
from .topk import BatchTopkActivation, TopkActivation

ACTIVATIONS_MAP: dict[str, type[SaveableModule]] = {
    cls.__name__: cls
    for cls in [
        TopkActivation,
        BatchTopkActivation,
        ReLUActivation,
        JumpReLUActivation,
    ]
}
