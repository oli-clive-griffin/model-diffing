from pathlib import Path

import wandb

from crosscoding.saveable_module import SaveableModule
from crosscoding.trainers.config_common import WandbConfig


def upload_experiment_checkpoint(
    model_checkpoint_path: str,
    previous_run_id: str,
    wandb_cfg: WandbConfig,
    step: int,
    epoch: int,
) -> None:
    """
    Utility function to upload a local checkpoint to wandb.
    """
    artifact = create_checkpoint_artifact(model_checkpoint_path, previous_run_id, step, epoch)

    previous_run = wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        id=previous_run_id,
        resume="allow",
    )

    previous_run.log_artifact(artifact)
    previous_run.finish()


def create_checkpoint_artifact(
    model_checkpoint_path: Path | str,
    run_id: str,
    step: int,
    epoch: int,
) -> wandb.Artifact:
    model_pt_path = Path(model_checkpoint_path) / SaveableModule.STATE_DICT_FNAME
    model_config_path = Path(model_checkpoint_path) / SaveableModule.MODEL_CFG_FNAME
    exp_config_path = Path(model_checkpoint_path).parent / "experiment_config.yaml"

    assert model_pt_path.exists(), f"Model file {model_pt_path} does not exist."
    assert model_config_path.exists(), f"Model config file {model_config_path} does not exist."
    assert exp_config_path.exists(), f"Experiment config file {exp_config_path} does not exist."

    name = f"{checkpoint_name(run_id)}"  # names must be unique within projects
    artifact = wandb.Artifact(name=name, type="model", metadata={"step": step, "epoch": epoch})
    artifact.add_dir(str(model_checkpoint_path), name="model")
    artifact.add_file(str(exp_config_path), name="experiment_config.yaml")
    return artifact


def download_experiment_checkpoint(
    run_id: str,
    version: str,
    destination_dir: Path | str,
    entity: str,
    project: str,
) -> Path:
    api = wandb.Api()
    art = api.artifact(f"{entity}/{project}/{checkpoint_name(run_id)}:{version}")
    root = f"{destination_dir}/{checkpoint_name(run_id)}_{version}"
    art.download(root=root)
    return Path(root)


def checkpoint_name(run_id: str) -> str:
    return f"model-checkpoint_run-{run_id}"


if __name__ == "__main__":
    ...
    # Example usage:

    # upload_experiment_checkpoint(
    #     model_checkpoint_path=".checkpoints/jan_update_crosscoder_example_2025-02-19_18-39-32/epoch_0_step_2499",
    #     previous_run_id="48rqbqcm",
    #     wandb_cfg=WandbConfig(entity="your_team", project="your_project"),
    # )

    # OR

    # download_experiment_checkpoint("48rqbqcm")
