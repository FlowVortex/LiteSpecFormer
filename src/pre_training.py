import argparse
import os

import socket

from accelerate import Accelerator

from litespecformer.config import LiteSpecFormerConfig, LiteSpecFormerTrainingConfig
from litespecformer.trainer import LiteSpecFormerPreTrainer

import wandb

os.environ["WANDB_API_KEY"] = "wandb"

parser = argparse.ArgumentParser(description="LiteSpecFormer pre-training")

parser.add_argument(
    "--team_name",
    type=str,
    required=True,
    help="Weights & Biases team/entity name used for experiment logging.",
)
parser.add_argument(
    "--project_name",
    type=str,
    required=True,
    help="Weights & Biases project name for grouping related runs.",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Base name of the experiment run (seed will be appended).",
)
parser.add_argument(
    "--scenario_name",
    type=str,
    required=True,
    help="Scenario/group label for organizing runs in W&B (e.g., dataset or setting).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed for reproducibility. Default: 0.",
)
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="Path to a checkpoint directory/file to resume training from. Default: None.",
)

args = parser.parse_args()


def obj_to_dict(obj):
    """
    Convert a generic object to a dictionary by reading its `__dict__` and
    filtering out built-in/private attributes that start with `__`.
    """
    return {
        key: value
        for key, value in obj.__dict__.items()
        if not key.startswith(
            "__"
        )  # Filter built-in fields such as __class__, __module__, etc.
    }


# Build experiment configurations.
model_config = LiteSpecFormerConfig()
forecasting_config = model_config.forecasting_config
training_config = LiteSpecFormerTrainingConfig()

# Merge all configs into one dictionary for W&B logging.
config_dict = {
    **obj_to_dict(model_config),
    **obj_to_dict(forecasting_config),
    **obj_to_dict(training_config),
}

# Create an Accelerator instance for distributed/mixed-precision training.
accelerator = Accelerator(
    device_placement=True,
    split_batches=True,
    mixed_precision="fp16",
    gradient_accumulation_steps=training_config.gradient_accumulation_steps,
)

if accelerator.is_main_process:
    # Initialize Weights & Biases tracking only on the main process.
    wandb.init(
        project=args.project_name,
        entity=args.team_name,
        notes=socket.gethostname(),
        name=args.experiment_name + "_" + str(args.seed),
        group=args.scenario_name,
        dir=str("./results"),
        job_type="training",
    )

    # Log the merged configuration dictionary to W&B.
    wandb.config.update(config_dict)

# Create the LiteSpecFormer pre-trainer.
trainer = LiteSpecFormerPreTrainer(
    accelerator=accelerator,
    model_config=model_config,
    training_config=training_config,
    resume_from_checkpoint=args.resume_from_checkpoint,
    model_params_only=True,
)

# Start the training loop.
trainer.run()
