from pathlib import Path

import wandb

from model_diffing.scripts.config_common import WandbConfig


def upload_experiment_checkpoint(model_checkpoint_path: str, previous_run_id: str, wandb_cfg: WandbConfig) -> None:
    artifact = create_checkpoint_artifact(model_checkpoint_path, previous_run_id)

    previous_run = wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        id=previous_run_id,
        resume="allow",
    )

    previous_run.log_artifact(artifact)
    previous_run.finish()


def create_checkpoint_artifact(model_checkpoint_path: Path | str, run_id: str) -> wandb.Artifact:
    model_pt_path = Path(model_checkpoint_path) / "model.pt"
    model_config_path = Path(model_checkpoint_path) / "model_cfg.yaml"
    exp_config_path = Path(model_checkpoint_path).parent / "config.yaml"

    assert model_pt_path.exists(), f"Model file {model_pt_path} does not exist."
    assert model_config_path.exists(), f"Model config file {model_config_path} does not exist."
    assert exp_config_path.exists(), f"Experiment config file {exp_config_path} does not exist."

    name = f"model-checkpoint_run-{run_id}"  # names must be unique within projects
    artifact = wandb.Artifact(name=name, type="model")
    artifact.add_dir(str(model_checkpoint_path), name="model")
    artifact.add_file(str(exp_config_path), name="experiment_config.yaml")
    return artifact


def download_experiment_checkpoint(
    run_id: str,
    version: str = "v0",
    destination_dir: str = "artifact_download",
    entity: str = "mars-model-diffing",
    project: str = "model-diffing",
) -> None:
    api = wandb.Api()
    art = api.artifact(f"{entity}/{project}/model-checkpoint_run-{run_id}:{version}")
    art.download(root=destination_dir)


if __name__ == "__main__":
    ...
    # Example usage:

    # upload_experiment_checkpoint(
    #     model_checkpoint_path=".checkpoints/jan_update_crosscoder_example_2025-02-19_18-39-32/epoch_0_step_2499",
    #     previous_run_id="48rqbqcm",
    #     wandb_cfg=WandbConfig(entity="mars-model-diffing", project="model-diffing"),
    # )

    # download_experiment_checkpoint("48rqbqcm")
