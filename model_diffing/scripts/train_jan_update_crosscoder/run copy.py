import os
from datetime import datetime
from pathlib import Path

import fire  # type: ignore
import yaml  # type: ignore

from model_diffing.log import logger
from model_diffing.scripts.base_acausal_trainer import save_config
from model_diffing.scripts.train_jan_update_crosscoder.config import JanUpdateExperimentConfig
from model_diffing.scripts.train_jan_update_crosscoder.run import build_jan_update_crosscoder_trainer


def main(config_path: Path):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config_path = Path(config_path)
    assert config_path.suffix == ".yaml", f"Config file {config_path} must be a YAML file."
    assert Path(config_path).exists(), f"Config file {config_path} does not exist."
    logger.info("Loading config...")
    for lr in [
        0.0006,
        0.0003,
        0.0001,
        0.00006,
        0.00003,
        0.00001,
    ]:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config_dict["train"]["optimizer"]["learning_rate"] = lr
        config = JanUpdateExperimentConfig(**config_dict)
        logger.info(f"Loaded config:\n{config.model_dump_json(indent=2)}")
        config.experiment_name += f"_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        logger.info(f"over-wrote experiment_name: {config.experiment_name}")
        logger.info(f"saving in save_dir: {config.save_dir}")
        save_config(config)
        logger.info("Building trainer")
        trainer = build_jan_update_crosscoder_trainer(config)
        logger.info("Training")
        trainer.train()


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(main)
