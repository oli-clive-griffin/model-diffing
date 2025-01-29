import torch
from torch.nn.utils import clip_grad_norm_

from model_diffing.scripts.train_l1_crosscoder.config import L1TrainConfig
from model_diffing.scripts.trainer import BaseTrainer
from model_diffing.utils import (
    calculate_explained_variance_ML,
    calculate_reconstruction_loss,
    get_explained_var_dict,
    l0_norm,
    multi_reduce,
    sparsity_loss_l1_of_norms,
)


class L1CrosscoderTrainer(BaseTrainer[L1TrainConfig]):
    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMLD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BMLD)
        sparsity_loss = sparsity_loss_l1_of_norms(self.crosscoder.W_dec_HMLD, train_res.hidden_BH)
        l1_coef = self._l1_coef_scheduler()
        loss = reconstruction_loss + l1_coef * sparsity_loss

        # backward
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        # metrics
        mean_l0 = multi_reduce(train_res.hidden_BH, "batch hidden", ("hidden", l0_norm), ("batch", torch.mean)).item()
        explained_variance_ML = calculate_explained_variance_ML(batch_BMLD, train_res.reconstructed_acts_BMLD)

        log_dict = {
            "train/l1_coef": l1_coef,
            "train/mean_l0": mean_l0,
            "train/mean_l0_pct": mean_l0 / self.crosscoder.hidden_dim,
            "train/reconstruction_loss": reconstruction_loss.item(),
            "train/sparsity_loss": sparsity_loss.item(),
            "train/loss": loss.item(),
            **get_explained_var_dict(explained_variance_ML, self.layers_to_harvest),
        }

        return log_dict

    def _l1_coef_scheduler(self) -> float:
        if self.step < self.cfg.l1_coef_n_steps:
            return self.cfg.l1_coef_max * self.step / self.cfg.l1_coef_n_steps
        else:
            return self.cfg.l1_coef_max
