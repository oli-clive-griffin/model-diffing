from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from model_diffing.models.activations.topk import TopkActivation
from model_diffing.scripts.base_trainer import BaseModelHookpointTrainer
from model_diffing.scripts.config_common import BaseTrainConfig
from model_diffing.scripts.utils import create_cosine_sim_and_relative_norm_histograms
from model_diffing.utils import calculate_explained_variance_X, calculate_reconstruction_loss, get_explained_var_dict


class TopKTrainer(BaseModelHookpointTrainer[BaseTrainConfig, TopkActivation]):
    def _train_step(self, batch_BMPD: torch.Tensor) -> None:
        self.optimizer.zero_grad()

        # fwd
        train_res = self.crosscoder.forward_train(batch_BMPD)

        # losses
        reconstruction_loss = calculate_reconstruction_loss(batch_BMPD, train_res.output_BXD)

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
            and (self.step + 1) % self.cfg.log_every_n_steps == 0
        ):
            explained_variance_dict = get_explained_var_dict(
                calculate_explained_variance_X(batch_BMPD, train_res.output_BXD),
                ("model", list(range(self.n_models))),
                ("hookpoint", self.hookpoints),
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
                crosscoding_dims = W_dec_HXD.shape[1:-1]
                assert crosscoding_dims == (self.n_models, self.n_hookpoints)
                log_dict.update(
                    create_cosine_sim_and_relative_norm_histograms(
                        W_dec_HMPD=W_dec_HXD,
                        hookpoints=self.hookpoints,
                    )
                )

            self.wandb_run.log(log_dict, step=self.step)
