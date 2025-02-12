from collections.abc import Iterator
from math import prod

import fire  # type: ignore
import torch
from einops import rearrange
from tqdm import tqdm  # type: ignore

from model_diffing.data.model_hookpoint_dataloader import BaseModelHookpointActivationsDataloader, build_dataloader
from model_diffing.log import logger
from model_diffing.models.activations import JumpReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder, InitStrategy
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.llms import build_llms
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateExperimentConfig, JumpReLUConfig
from model_diffing.scripts.train_jan_update_crosscoder.trainer import JanUpdateCrosscoderTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device, inspect, round_up


def build_jan_update_crosscoder_trainer(cfg: JanUpdateExperimentConfig) -> JanUpdateCrosscoderTrainer:
    device = get_device()

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
        batch_size=cfg.train.batch_size,
        cache_dir=cfg.cache_dir,
        device=device,
    )

    n_models = len(llms)
    n_hookpoints = len(cfg.hookpoints)

    crosscoder = _build_jan_update_crosscoder(
        n_models=n_models,
        n_hookpoints=n_hookpoints,
        d_model=llms[0].cfg.d_model,
        cc_hidden_dim=cfg.crosscoder.hidden_dim,
        jumprelu=cfg.crosscoder.jumprelu,
        data_loader=dataloader,
        device=device,
    )
    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg) if cfg.wandb else None

    return JanUpdateCrosscoderTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        hookpoints=cfg.hookpoints,
        experiment_name=cfg.experiment_name,
    )


def _build_jan_update_crosscoder(
    n_models: int,
    n_hookpoints: int,
    d_model: int,
    cc_hidden_dim: int,
    jumprelu: JumpReLUConfig,
    data_loader: BaseModelHookpointActivationsDataloader,
    device: torch.device,  # for computing b_enc
) -> AcausalCrosscoder[JumpReLUActivation]:
    cc = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_hookpoints),
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        init_strategy=JanUpdateInitStrategy(
            activations_iterator_BXD=data_loader.get_shuffled_activations_iterator_BMPD(),
            device=device,
            n_examples_to_sample=100_000,
            firing_sparsity=10_000,
        ),
        hidden_activation=JumpReLUActivation(
            size=cc_hidden_dim,
            bandwidth=jumprelu.bandwidth,
            threshold_init=jumprelu.threshold_init,
            backprop_through_input=jumprelu.backprop_through_jumprelu_input,
        ),
    )

    cc.to(device)

    return cc


class JanUpdateInitStrategy(InitStrategy[JumpReLUActivation]):
    def __init__(
        self,
        activations_iterator_BXD: Iterator[torch.Tensor],
        device: torch.device,
        n_examples_to_sample: int = 100_000,
        firing_sparsity: float = 10_000,
    ):
        self.activations_iterator_BXD = activations_iterator_BXD
        self.device = device
        self.n_examples_to_sample = n_examples_to_sample
        self.firing_sparsity = firing_sparsity

    def init_weights(self, cc: AcausalCrosscoder[JumpReLUActivation]) -> None:
        n = prod(cc.crosscoding_dims) * cc.d_model
        m = cc.hidden_dim

        cc.W_dec_HXD.uniform_(-1.0 / n, 1.0 / n)
        cc.W_enc_XDH.copy_(
            rearrange(cc.W_dec_HXD, "hidden ... -> ... hidden")  #
            * (n / m)
        )

        calibrated_b_enc_H = self.compute_b_enc_H(
            cc.W_enc_XDH,
            cc.hidden_activation.log_threshold_H.exp(),
        )

        cc.b_enc_H.copy_(calibrated_b_enc_H)

        cc.b_dec_XD.zero_()

    def compute_b_enc_H(self, W_enc_XDH: torch.Tensor, initial_jumprelu_threshold_H: torch.Tensor) -> torch.Tensor:
        logger.info(f"Harvesting pre-bias for {self.n_examples_to_sample} examples")

        pre_bias_NH = self.harvest_pre_bias_NH(W_enc_XDH)

        # find the threshold for each idx H such that 1/10_000 of the examples are above the threshold
        quantile_H = torch.quantile(pre_bias_NH, 1 - 1 / self.firing_sparsity, dim=0)

        # firing is when the post-bias is above the jumprelu threshold therefore, we subtract
        # the quantile from the initial jumprelu threshold, such the 1/firing_sparsity of the
        # examples are above the threshold
        b_enc_H = initial_jumprelu_threshold_H - quantile_H

        logger.info(f"computed b_enc_H. Sample: {b_enc_H[:10]}. mean: {b_enc_H.mean()}, std: {b_enc_H.std()}")

        return b_enc_H

    def harvest_pre_bias_NH(self, W_enc_XDH: torch.Tensor) -> torch.Tensor:
        def get_batch_pre_bias() -> torch.Tensor:
            # this is essentially the first step of the crosscoder forward pass, but not worth
            # creating a new method for it, just (easily) reimplementing it here
            batch_BXD = next(self.activations_iterator_BXD)
            x_BH = torch.einsum("b ... d, ... d h -> b h", batch_BXD, W_enc_XDH)
            return x_BH

        sample_BH = get_batch_pre_bias()
        batch_size, hidden_size = sample_BH.shape

        rounded_n_examples_to_sample = round_up(self.n_examples_to_sample, to_multiple_of=batch_size)

        if rounded_n_examples_to_sample > self.n_examples_to_sample:
            logger.warning(
                f"rounded n_examples_to_sample from {self.n_examples_to_sample} to {rounded_n_examples_to_sample} "
                f"to be divisible by the batch size {batch_size}"
            )

        num_batches = rounded_n_examples_to_sample // batch_size

        pre_bias_buffer_NH = torch.empty(rounded_n_examples_to_sample, hidden_size, device=self.device)
        logger.info(f"pre_bias_buffer_NH: {inspect(pre_bias_buffer_NH)}")

        pre_bias_buffer_NH[:batch_size] = sample_BH

        for i in tqdm(
            range(1, num_batches), desc="Harvesting pre-bias"
        ):  # start at 1 because we already sampled the first batch
            batch_pre_bias_BH = get_batch_pre_bias()
            pre_bias_buffer_NH[batch_size * i : batch_size * (i + 1)] = batch_pre_bias_BH

        return pre_bias_buffer_NH


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_jan_update_crosscoder_trainer, JanUpdateExperimentConfig))
