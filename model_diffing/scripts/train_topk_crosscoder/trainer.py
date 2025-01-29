import torch
from torch.nn.utils import clip_grad_norm_

from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.trainer import BaseTrainer
from model_diffing.utils import (
    calculate_explained_variance_ML,
    calculate_reconstruction_loss,
    get_explained_var_dict,
)


class TopKTrainer(BaseTrainer[BaseTrainConfig]):
    def _train_step(self, batch_BMLD: torch.Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMLD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BMLD)

        # backward
        reconstruction_loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        # metrics
        explained_variance_ML = calculate_explained_variance_ML(batch_BMLD, train_res.reconstructed_acts_BMLD)

        # is measuring l0 meaningful here? I don't think so in the case of topk

        log_dict = {
            "train/reconstruction_loss": reconstruction_loss.item(),
            **get_explained_var_dict(explained_variance_ML, self.layers_to_harvest),
        }

        return log_dict
