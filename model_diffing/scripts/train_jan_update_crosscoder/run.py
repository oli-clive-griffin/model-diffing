from pathlib import Path

import fire  # type: ignore
import torch
import yaml  # type: ignore
from einops import rearrange

from model_diffing.dataloader.activations import BaseActivationsDataloader
from model_diffing.dataloader.data import build_dataloader
from model_diffing.log import logger
from model_diffing.models.crosscoder import (
    AcausalCrosscoder,
    JumpReLUActivation,
    build_jumprelu_crosscoder,
)
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateExperimentConfig, JumpReLUConfig
from model_diffing.scripts.train_jan_update_crosscoder.trainer import JanUpdateCrosscoderTrainer
from model_diffing.utils import build_wandb_run, get_device, size_GB


def build_jan_update_crosscoder(
    n_models: int,
    n_layers: int,
    d_model: int,
    cc_hidden_dim: int,
    jumprelu: JumpReLUConfig,
    data_loader: BaseActivationsDataloader,
    device: torch.device,  # for computing b_enc
) -> AcausalCrosscoder[JumpReLUActivation]:
    cc = build_jumprelu_crosscoder(
        n_models=n_models,
        n_layers=n_layers,
        d_model=d_model,
        cc_hidden_dim=cc_hidden_dim,
        dec_init_norm=0,  # dec_init_norm doesn't matter here as we override weights below
        jumprelu=jumprelu,
    )

    with torch.no_grad():
        # parameters from the jan update doc

        n = float(n_models * n_layers * d_model)  # n is the size of the input space
        m = float(cc_hidden_dim)  # m is the size of the hidden space

        # W_dec ~ U(-1/n, 1/n) (from doc)
        cc.W_dec_HMLD.uniform_(-1.0 / n, 1.0 / n)

        # For now, assume we're in the X == Y case.
        # Therefore W_enc = (n/m) * W_dec^T
        cc.W_enc_MLDH.copy_(
            rearrange(cc.W_dec_HMLD, "hidden model layer d_model -> model layer d_model hidden")  #
            * (n / m)
        )

        calibrated_b_enc_H = _compute_b_enc_H(
            data_loader,
            cc.W_enc_MLDH,
            cc.hidden_activation.log_threshold_H.exp(),
            device,
        )
        cc.b_enc_H.copy_(calibrated_b_enc_H)

        # no data-dependent initialization of b_dec
        cc.b_dec_MLD.zero_()

    return cc


def _compute_b_enc_H(
    data_loader: BaseActivationsDataloader,
    W_enc_MLDH: torch.Tensor,
    initial_threshold_H: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int = 100_000,
) -> torch.Tensor:
    pre_bias_pre_act_buffer_NH = harvest_pre_pre_bias_acts(data_loader, W_enc_MLDH, device, n_examples_to_sample)

    # find the threshold for each idx H such that 1/10_000 of the examples are above the threshold
    quantile_H = torch.quantile(pre_bias_pre_act_buffer_NH, 1 - 1 / 10_000, dim=0)

    b_enc_H = initial_threshold_H - quantile_H

    return b_enc_H


def harvest_pre_pre_bias_acts(
    data_loader: BaseActivationsDataloader,
    W_enc_MLDH: torch.Tensor,
    device: torch.device,
    n_examples_to_sample: int = 100_000,
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

    activations_iterator_BMLD = data_loader.get_shuffled_activations_iterator_BMLD()

    def get_batch_pre_bias_pre_act() -> torch.Tensor:
        # this is essentially the first step of the crosscoder forward pass, but not worth
        # creating a new method for it, just (easily) reimplementing it here
        batch_BMLD = next(activations_iterator_BMLD)
        x_BH = torch.einsum("b m l d, m l d h -> b h", batch_BMLD, W_enc_MLDH)
        return x_BH

    first_sample_BH = get_batch_pre_bias_pre_act()
    hidden_size = first_sample_BH.shape[1]

    pre_bias_pre_act_buffer_NH = torch.empty(n_examples_to_sample, hidden_size, device=device)
    logger.info(
        f"pre_bias_pre_act_buffer_NH.shape: {pre_bias_pre_act_buffer_NH.shape}, "
        f"size: {size_GB(pre_bias_pre_act_buffer_NH)} GB"
    )

    pre_bias_pre_act_buffer_NH[:batch_size] = first_sample_BH
    examples_sampled = batch_size

    while examples_sampled < n_examples_to_sample:
        batch_pre_bias_pre_act_BH = get_batch_pre_bias_pre_act()
        pre_bias_pre_act_buffer_NH[examples_sampled : examples_sampled + batch_size] = batch_pre_bias_pre_act_BH
        examples_sampled += batch_size
    return pre_bias_pre_act_buffer_NH


def build_jan_update_crosscoder_trainer(cfg: JanUpdateExperimentConfig) -> JanUpdateCrosscoderTrainer:
    device = get_device()

    dataloader = build_dataloader(cfg.data, cfg.cache_dir, device)
    (_, n_layers, n_models, d_model) = dataloader.batch_shape_BMLD()

    crosscoder = build_jan_update_crosscoder(
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


def load_config(config_path: Path) -> JanUpdateExperimentConfig:
    """Load the config from a YAML file into a Pydantic model."""
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = JanUpdateExperimentConfig(**config_dict)
    return config


def main(config_path: str) -> None:
    logger.info("Loading config...")
    config = load_config(Path(config_path))
    logger.info("Loaded config")
    trainer = build_jan_update_crosscoder_trainer(config)
    trainer.train()


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(main)
