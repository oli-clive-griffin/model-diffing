import fire  # type: ignore

from model_diffing.data.model_layer_dataloader import build_dataloader
from model_diffing.log import logger
from model_diffing.models.activations.topk import TopkActivation
from model_diffing.models.crosscoder import AcausalCrosscoder
from model_diffing.scripts.base_trainer import run_exp
from model_diffing.scripts.train_topk_crosscoder.config import TopKExperimentConfig
from model_diffing.scripts.train_topk_crosscoder.trainer import TopKTrainer
from model_diffing.scripts.utils import build_wandb_run
from model_diffing.utils import get_device


def build_trainer(cfg: TopKExperimentConfig) -> TopKTrainer:
    device = get_device()

    dataloader = build_dataloader(cfg.data, cfg.train.batch_size, cfg.cache_dir, device)
    _, n_models, n_layers, d_model = dataloader.batch_shape_BMLD()

    crosscoder = AcausalCrosscoder(
        crosscoding_dims=(n_models, n_layers),
        d_model=d_model,
        hidden_dim=cfg.crosscoder.hidden_dim,
        dec_init_norm=cfg.crosscoder.dec_init_norm,
        hidden_activation=TopkActivation(k=cfg.crosscoder.k),
    )

    crosscoder = crosscoder.to(device)

    wandb_run = build_wandb_run(cfg) if cfg.wandb else None

    return TopKTrainer(
        cfg=cfg.train,
        activations_dataloader=dataloader,
        crosscoder=crosscoder,
        wandb_run=wandb_run,
        device=device,
        layers_to_harvest=cfg.data.activations_harvester.layer_indices_to_harvest,
        experiment_name=cfg.experiment_name,
    )


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(run_exp(build_trainer, TopKExperimentConfig))
