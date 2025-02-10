from model_diffing.utils import SaveableModule

from .jumprelu import JumpReLUActivation
from .relu import ReLUActivation
from .topk import BatchTopkActivation, TopkActivation

ACTIVATIONS: dict[str, type[SaveableModule]] = {
    "TopkActivation": TopkActivation,
    "BatchTopkActivation": BatchTopkActivation,
    "ReLU": ReLUActivation,
    "JumpReLU": JumpReLUActivation,
}
