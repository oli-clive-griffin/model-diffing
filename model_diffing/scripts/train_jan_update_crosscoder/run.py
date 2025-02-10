import fire  # type: ignore
import torch
from einops import rearrange
from tqdm import tqdm  # type: ignore

from model_diffing.data.model_layer_dataloader import BaseModelLayerActivationsDataloader, build_dataloader
from model_diffing.log import logger
from model_diffing.models.activations import JumpReLUActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateExperimentConfig, JumpReLUConfig
from model_diffing.scripts.train_jan_update_crosscoder.trainer import JanUpdateCrosscoderTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device, inspect


def build_jan_update_crosscoder_trainer(cfg: JanUpdateExperimentConfig) -> JanUpdateCrosscoderTrainer:
    device = get_device()

    dataloader = build_dataloader(cfg.data, cfg.train.batch_size, cfg.cache_dir, device)
    _, n_models, n_layers, d_model = dataloader.batch_shape_BMLD()

    crosscoder = _build_jan_update_crosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
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
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
        experiment_name=cfg.experiment_name,
    )


def _build_jan_update_crosscoder(
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    jumprelu: JumpReLUConfig,
    data_loader: BaseModelLayerActivationsDataloader,
    device: torch.device,  # for computing b_enc
) -> AcausalCrosscoder[JumpReLUActivation]:
    cc = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_layers),
        d_model=d_model,
        hidden_dim=cc_hidden_dim,
        dec_init_norm=0,  # dec_init_norm doesn't matter here as we override weights below
        hidden_activation=JumpReLUActivation(
            size=cc_hidden_dim,
            bandwidth=jumprelu.bandwidth,
            threshold_init=jumprelu.threshold_init,
            backprop_through_input=jumprelu.backprop_through_jumprelu_input,
        ),
    )

    cc.to(device)

    with torch.no_grad():
        # parameters from the jan update doc

        n = float(n_models * n_layers * d_model)  # n is the size of the input space
        m = float(cc_hidden_dim)  # m is the size of the hidden space

        # W_dec ~ U(-1/n, 1/n) (from doc)
        cc.W_dec_HXD.uniform_(-1.0 / n, 1.0 / n)

        # For now, assume we're in the X == Y case.
        # Therefore W_enc = (n/m) * W_dec^T
        cc.W_enc_XDH.copy_(
            rearrange(cc.W_dec_HXD, "hidden ... -> ... hidden")  #
            * (n / m)
        )

        # raise ValueError("need to fix this: 10000/m, not 1/10000")
        calibrated_b_enc_H = _compute_b_enc_H(
            data_loader,
            cc.W_enc_XDH,
            cc.hidden_activation.log_threshold_H.exp(),
            device,
            n_examples_to_sample=50_000,
            firing_sparsity=4,
        )
        cc.b_enc_H.copy_(calibrated_b_enc_H)

        # no data-dependent initialization of b_dec
        cc.b_dec_XD.zero_()

    return cc


def _compute_b_enc_H(
    data_loader: BaseModelLayerActivationsDataloader,
    W_enc_MLDH: torch.Tensor,
    initial_jumprelu_threshold_H: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int = 100_000,
    firing_sparsity: float = 10_000,
) -> torch.Tensor:
    logger.info(f"Harvesting pre-bias for {n_examples_to_sample} examples")

    pre_bias_NH = _harvest_pre_bias_NH(data_loader, W_enc_MLDH, device, n_examples_to_sample)

    # find the threshold for each idx H such that 1/10_000 of the examples are above the threshold
    quantile_H = torch.quantile(pre_bias_NH, 1 - 1 / firing_sparsity, dim=0)

    # firing is when the post-bias is above the jumprelu threshold therefore, we subtract
    # the quantile from the initial jumprelu threshold, such the 1/firing_sparsity of the
    # examples are above the threshold
    b_enc_H = initial_jumprelu_threshold_H - quantile_H

    logger.info(f"computed b_enc_H. Sample: {b_enc_H[:10]}. mean: {b_enc_H.mean()}, std: {b_enc_H.std()}")

    return b_enc_H


def _harvest_pre_bias_NH(
    data_loader: BaseModelLayerActivationsDataloader,
    W_enc_MLDH: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int,
) -> torch.Tensor:
    batch_size = data_loader.batch_shape_BMLD()[0]

    remainder = n_examples_to_sample % batch_size
    if remainder != 0:
        logger.warning(
            f"n_examples_to_sample {n_examples_to_sample} must be divisible by the batch "
            f"size {batch_size}. Rounding up to the nearest multiple of batch_size."
        )
        # Round up to the nearest multiple of batch_size:
        n_examples_to_sample = (((n_examples_to_sample - remainder) // batch_size) + 1) * batch_size

        logger.info(f"n_examples_to_sample is now {n_examples_to_sample}")

    num_batches = n_examples_to_sample // batch_size

    activations_iterator_BMLD = data_loader.get_shuffled_activations_iterator_BMLD()

    def get_batch_pre_bias() -> torch.Tensor:
        # this is essentially the first step of the crosscoder forward pass, but not worth
        # creating a new method for it, just (easily) reimplementing it here
        batch_BMLD = next(activations_iterator_BMLD)
        x_BH = torch.einsum("b m l d, m l d h -> b h", batch_BMLD, W_enc_MLDH)
        return x_BH

    first_sample_BH = get_batch_pre_bias()
    hidden_size = first_sample_BH.shape[1]

    pre_bias_buffer_NH = torch.empty(n_examples_to_sample, hidden_size, device=device)
    logger.info(f"pre_bias_buffer_NH: {inspect(pre_bias_buffer_NH)}")

    pre_bias_buffer_NH[:batch_size] = first_sample_BH

    for i in tqdm(
        range(1, num_batches), desc="Harvesting pre-bias"
    ):  # start at 1 because we already sampled the first batch
        batch_pre_bias_BH = get_batch_pre_bias()
        pre_bias_buffer_NH[batch_size * i : batch_size * (i + 1)] = batch_pre_bias_BH

    return pre_bias_buffer_NH


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_jan_update_crosscoder_trainer, JanUpdateExperimentConfig))
