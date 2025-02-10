from pathlib import Path

import fire  # type: ignore
import wandb

from model_diffing.log import logger

# usage:
# python model_diffing/scripts/upload_model_to_wandb/main.py --model_checkpoint .checkpoints/path/to/checkpoint.pt --previous_run_id <your run id> (something something like 6mj8s5oy, not the run name)


def main(model_checkpoint: str, previous_run_id: str) -> None:
    wandb.init(project="model-diffing", entity="mars-model-diffing")
    logger.info("Loading model checkpoint...")
    model_checkpoint_path = Path(model_checkpoint)
    assert model_checkpoint_path.exists(), f"Model checkpoint {model_checkpoint_path} does not exist."
    logger.info("Loaded model checkpoint")
    previous_run = wandb.Api().run(f"mars-model-diffing/model-diffing/{previous_run_id}")
    artifact = wandb.Artifact(name="model-checkpoint", type="model")
    artifact.add_file("path/to/local/checkpoint.pt")
    wandb.log_artifact(artifact)
    previous_run.log_artifact(artifact)


if __name__ == "__main__":
    logger.info("Starting...")
    fire.Fire(main)
