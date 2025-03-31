import torch


class FiringTracker:
    def __init__(self, activation_size: int, device: torch.device):
        self._activation_size = activation_size
        self.tokens_since_fired_L = torch.zeros(activation_size, dtype=torch.int64, device=device)

    def add_batch(self, latents_BL: torch.Tensor) -> None:
        batch_size = latents_BL.shape[0]
        firing_L = latents_BL.any(dim=0).detach()
        self.tokens_since_fired_L[firing_L] = 0
        self.tokens_since_fired_L[~firing_L] += batch_size
