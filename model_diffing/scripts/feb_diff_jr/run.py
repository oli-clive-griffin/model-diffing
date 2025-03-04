from typing import TypeVar

import fire  # type: ignore
import torch  # type: ignore
from einops import rearrange

from model_diffing.data.model_hookpoint_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models.activations.jumprelu import AnthropicJumpReLUActivation
from model_diffing.models.diffing_crosscoder import N_MODELS, DiffingCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.feb_diff_jr.config import JumpReLUModelDiffingFebUpdateExperimentConfig
from model_diffing.scripts.feb_diff_jr.trainer import ModelDiffingFebUpdateJumpReLUTrainer
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_jan_update_crosscoder.run import BaseJumpReLUInitStrategy
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import SaveableModule, get_device

TActivation = TypeVar("TActivation", bound=SaveableModule)


def build_feb_update_crosscoder_trainer(
    cfg: JumpReLUModelDiffingFebUpdateExperimentConfig,
) -> ModelDiffingFebUpdateJumpReLUTrainer:
    device = get_device()

    assert len(cfg.data.activations_harvester.llms) == 2, "we only support 2 models for now"

    llms = build_llms(
        cfg.data.activations_harvester.llms,
        cfg.cache_dir,
        device,
        dtype=cfg.data.activations_harvester.inference_dtype,
    )

    dataloader = build_dataloader(
        cfg=cfg.data,
        llms=llms,
        hookpoints=cfg.hookpoints,
        batch_size=cfg.train.minibatch_size(),
        cache_dir=cfg.cache_dir,
    )

    crosscoder = DiffingCrosscoder(
        d_model=llms[0].cfg.d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        n_explicitly_shared_latents=cfg.crosscoder.n_shared_latents,
        init_strategy=ModelDiffingDataDependentJumpReLUInitStrategy(
            activations_iterator_BXD=dataloader.get_activations_iterator_BMPD(),
            initial_approx_firing_pct=cfg.crosscoder.initial_approx_firing_pct,
            n_tokens_for_threshold_setting=cfg.crosscoder.n_tokens_for_threshold_setting,
        ),
        hidden_activation=AnthropicJumpReLUActivation(
            size=cfg.crosscoder.hidden_dim,
            bandwidth=cfg.crosscoder.jumprelu.bandwidth,
            log_threshold_init=cfg.crosscoder.jumprelu.log_threshold_init,
            backprop_through_input=cfg.crosscoder.jumprelu.backprop_through_jumprelu_input,
        ),
    )
    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg)

    return ModelDiffingFebUpdateJumpReLUTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        save_dir=cfg.save_dir,
    )


class ModelDiffingDataDependentJumpReLUInitStrategy(
    BaseJumpReLUInitStrategy[DiffingCrosscoder[AnthropicJumpReLUActivation]]
):
    """Implementation for DiffingCrosscoder models."""

    @torch.no_grad()
    def init_weights(self, cc: DiffingCrosscoder[AnthropicJumpReLUActivation]) -> None:
        n = cc.d_model * N_MODELS
        m = cc.hidden_dim

        cc._W_dec_shared_HsD.uniform_(-1.0 / n, 1.0 / n)
        cc._W_dec_indep_HiMD.uniform_(-1.0 / n, 1.0 / n)

        cc.W_enc_MDH.copy_(
            rearrange(cc.theoretical_decoder_W_dec_HMD(), "hidden ... -> ... hidden")  #
            * (n / m)
        )

        cc.b_enc_H.copy_(self.get_calibrated_b_enc_H(cc.W_enc_MDH, cc.hidden_activation))
        cc.b_dec_MD.zero_()


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_feb_update_crosscoder_trainer, JumpReLUModelDiffingFebUpdateExperimentConfig))
