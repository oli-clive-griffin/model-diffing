from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.topk import TopkActivation
from model_diffing.scripts.base_trainer import BaseModelLayerTrainer
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.utils import create_cosine_sim_and_relative_norm_histograms
from model_diffing.utils import calculate_explained_variance_X, calculate_reconstruction_loss, get_explained_var_dict


class TopKTrainer(BaseModelLayerTrainer[BaseTrainConfig, TopkActivation]):
    def _train_step(self, batch_BMLD: torch.Tensor) -> None:
        self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMLD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMLD, train_res.reconstructed_acts_BXD)

        # backward
        reconstruction_loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        self.optimizer.step()
        assert len(self.optimizer.param_groups) == 1, "sanity check failed"
        self.optimizer.param_groups[0]["lr"] = self.lr_scheduler(self.step)

        # is measuring l0 meaningful here? I don't think so in the case of topk
        if (
            self.wandb_run is not None
            and self.cfg.log_every_n_steps is not None
            and self.step % self.cfg.log_every_n_steps == 0
        ):
            explained_variance_dict = get_explained_var_dict(
                calculate_explained_variance_X(batch_BMLD, train_res.reconstructed_acts_BXD),
                ("model", list(range(self.n_models))),
                ("layer", self.layers_to_harvest),
            )

            log_dict: dict[str, Any] = {
                "train/epoch": self.epoch,
                "train/unique_tokens_trained": self.unique_tokens_trained,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/reconstruction_loss": reconstruction_loss.item(),
                **explained_variance_dict,
            }

            if self.n_models == 2:
                W_dec_HXD = self.crosscoder.W_dec_HXD.detach().cpu()
                assert W_dec_HXD.shape[1:-1] == (self.n_models, self.n_layers)
                log_dict.update(
                    create_cosine_sim_and_relative_norm_histograms(
                        W_dec_HMLD=W_dec_HXD,
                        layers=self.layers_to_harvest,
                    )
                )

            self.wandb_run.log(log_dict, step=self.step)
