import torch

from crosscode.utils import l2_norm


@torch.no_grad()
def random_direction_init_(tensor: torch.Tensor, norm: float) -> None:
    tensor.normal_()
    tensor.div_(l2_norm(tensor, dim=-1, keepdim=True))
    tensor.mul_(norm)