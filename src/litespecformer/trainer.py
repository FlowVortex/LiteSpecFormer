from typing import (
    TYPE_CHECKING,
    cast,
    Optional,
    Union,
    Tuple,
    Dict,
    List,
    Iterator,
    Any,
)

from datetime import datetime
import os
from os import path
from time import sleep

from accelerate import Accelerator
from colorama import Fore, Style
from datasets import Dataset as HFDataset, concatenate_datasets

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from torch.utils.data import Dataset, IterableDataset, DataLoader

from tqdm import tqdm

from safetensors.torch import load_file

from litespecformer.config import (
    LiteSpecFormerConfig,
    LiteSpecFormerForecastingConfig,
    LiteSpecFormerTrainingConfig,
)
from litespecformer.dataset import (
    LiteSpecFormerDataset,
    LiteSpecFormerTestingDataset,
)
from litespecformer.model import LiteSpecFormerModel
from litespecformer.utils import (
    plot_confidence_prediction,
    count_parameters,
    cyclic_sample_generator,
)

import wandb


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers to ensure reproducibility.
    The seed is derived from the initial seed of the main process.

    Parameters
    ----------
    worker_id
        The worker ID provided by DataLoader.

    Returns
    -------
        None
    """
    import random

    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LiteSpecFormerPreTrainer(object):
    """
    Trainer for pre-training `LiteSpecFormer` as a Time Series Foundation Model.

    This trainer implements next-token forecasting pre-training: the model learns to
    predict future time-series patches from a given context window. Training is
    orchestrated through Hugging Face Accelerate for distributed and mixed-precision
    execution, with periodic evaluation, checkpoint saving, and Weights & Biases logging.

    Training data is loaded from disk in subsets; subsets are cycled across processes
    so that each GPU/TPU worker trains on a different shard. The training subset is
    refreshed every ``num_subset_steps`` steps. Evaluation runs every
    ``num_test_steps`` steps on a held-out test set under multiple context lengths.

    Call :meth:`run` to start the full pre-training loop (train, evaluate, save, visualize).

    Parameters
    ----------
    accelerator
        Hugging Face :class:`~accelerate.Accelerator` instance for distributed
        training, device placement, and gradient synchronization.
    model_config
        Model architecture and forecasting-task configuration for `LiteSpecFormer`.
    training_config
        Hyperparameters and paths for the pre-training run (learning rate, batch size,
        data paths, checkpoint directory, evaluation intervals, etc.).
    resume_from_checkpoint
        Optional path to a checkpoint directory to resume training from. When set,
        training state (or model weights only) is restored before the loop starts.
    model_params_only
        If ``True`` and ``resume_from_checkpoint`` is set, load only model weights
        from ``model.safetensors`` and reinitialize the optimizer and scheduler.
        If ``False``, restore the full training state via Accelerate.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        model_config: LiteSpecFormerConfig,
        training_config: LiteSpecFormerTrainingConfig,
        resume_from_checkpoint: Optional[str] = None,
        model_params_only: Optional[bool] = False,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config

        # Hyperparameters related to prediction task configuration
        self.forecasting_config: LiteSpecFormerForecastingConfig = (
            model_config.forecasting_config
        )

        # The accelerator for distributed training
        self.accelerator = accelerator

        # Load the model to be pre-trained
        self.model = self.get_model()

        if self.accelerator.is_main_process:
            # Count the number of model params
            count_parameters(model=self.model)

        # Load the optimizer and scheduler for model pre-training
        self.optimizer, self.scheduler = self.get_optimizer(
            params=self.get_trainable_params()
        )

        # Total number of training rounds required
        self.num_training_steps = self.training_config.num_training_steps

        # Model validation will be performed every num_test_steps steps
        self.num_test_steps = self.training_config.num_test_steps

        # The training dataset will be switched every num_subset_steps steps
        # to ensure the model can be trained on the whole dataset
        self.num_subset_steps = self.training_config.num_training_steps

        # Record the name of this experiment configuration.
        self.setting = f"LiteSpecFormer_{datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')}_bs{self.training_config.batch_size}_lr{self.training_config.learning_rate}_w{self.training_config.weight_decay}_b1{self.training_config.beta1}_b2{self.training_config.beta2}_eps{self.training_config.epsilon}_warmup{self.training_config.num_warmup_steps}"

        # Save the address of the model breakpoint
        if self.accelerator.is_main_process:
            self.checkpoint_dir = path.join(
                training_config.checkpoint_path,
                self.setting,
            )
            self.accelerator.print(
                f"Created checkpoints to {self.checkpoint_dir}", end=" -> "
            )
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.accelerator.print("Done")

        # The root path for the training and testing datasets
        self.data_path = training_config.data_path
        self.train_data_path = path.join(self.data_path, "train")
        self.test_data_path = path.join(self.data_path, "test")

        # Get the number of subset in the training dataset
        self.training_subset = [
            path.join(self.train_data_path, subset)
            for subset in os.listdir(self.train_data_path)
        ]
        self.num_training_subset = len(os.listdir(self.train_data_path))

        # Get the generator for cyclic sampling of training subsets
        self.training_subset_generator: Iterator[Tuple[Any, ...]] = (
            cyclic_sample_generator(
                data_list=self.training_subset,
                sample_size=self.accelerator.num_processes,
            )
        )

        # Different context lengths for testing
        self.eval_context_lengths = training_config.eval_context_lengths

        # The path to resume the checkpoint
        self.resume_from_checkpoint = resume_from_checkpoint

        # Whether to only load the model parameters from the checkpoint
        self.model_params_only = model_params_only

    @property
    def learning_rate(self) -> float:
        """Obtain the varying learning rate at different steps of model training."""
        return self.optimizer.param_groups[0]["lr"]

    @property
    def batch_size(self) -> int:
        """The batch size of model pre-training"""
        return self.training_config.train_batch_size * self.accelerator.num_processes

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Obtain trainable parameters of the model in experiment."""
        assert hasattr(
            self, "model"
        ), "The model has not been built yet. Please call the build_model method first."
        return [p for p in self.model.parameters() if p.requires_grad]

    def get_model(self) -> Union[nn.Module, LiteSpecFormerModel]:
        """Load the model to be Pre-trained."""

        self.accelerator.print(f"Loading model: {'LiteSpecFormer'}", end=" -> ")
        model = LiteSpecFormerModel(config=self.model_config)
        self.accelerator.print("Done")

        return model

    def get_optimizer(
        self, params: torch.Tensor
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
        """
        Obtain the optimizer needed for model training and the learning rate scheduler for dynamic learning rate adjustment.

        Parameters
        ----------
        params
            The trainable parameters of the model.

        Returns
        -------
        optimizer
            The optimizer for model training.
        scheduler
            The learning rate scheduler for dynamic learning rate adjustment.
        """

        # Create the optimizer for model training
        optimizer = optim.AdamW(
            params=params,
            lr=self.training_config.learning_rate,
            betas=(self.training_config.beta1, self.training_config.beta2),
            eps=self.training_config.epsilon,
            weight_decay=self.training_config.weight_decay,
        )

        # Create the learning rate scheduler for dynamic learning rate adjustment
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.training_config.learning_rate,
            total_steps=self.training_config.num_training_steps
            * self.accelerator.num_processes,
            pct_start=self.training_config.num_warmup_steps
            / self.training_config.num_training_steps,
            anneal_strategy=self.training_config.anneal_strategy,
        )

        return optimizer, scheduler

    def get_train_dataset(self, step: int) -> LiteSpecFormerDataset:
        """
        Load the dataset for model pre-training.

        Parameters
        ----------
        step
            The current training step, which can be used to determine when to switch the training dataset.

        Returns
        -------
        training_dataset
            The dataset for model pre-training at the current training step.
        """

        self.accelerator.print(
            f"Loading training dataset from {self.train_data_path}", end=" -> "
        )

        # Here, different datasets need to be loaded for different processes depending on the dataset generator.
        data_path_list = next(self.training_subset_generator)

        # Logging the information of the training dataset loaded for the current process
        self.accelerator.print("Currect data path:", data_path_list)

        # Load the training dataset from disk for the current process
        train_dataset = HFDataset.load_from_disk(
            dataset_path=data_path_list[self.accelerator.process_index],
            keep_in_memory=True,  # Keep the dataset in memory for faster access during training
        ).select_columns(["target"])
        train_dataset.set_format("torch")

        self.wait_for_everyone()

        # Only the main process is responsible for creating the training dataset and
        # logging the information of the loaded dataset to Weights & Biases,
        # while other processes are waiting for the main process to complete the dataset creation before proceeding with training.
        self.accelerator.print(
            Fore.RED + "Now is creating the LiteSpecFormer dataset..." + Style.RESET_ALL
        )

        # Building a dataset for training a base time series model
        training_dataset = LiteSpecFormerDataset(
            inputs=train_dataset,
            context_length=self.forecasting_config.context_length,
            prediction_length=self.forecasting_config.output_patch_size,
            batch_size=self.training_config.batch_size,
            output_patch_size=self.forecasting_config.output_patch_size,
            min_past=self.forecasting_config.min_past,
        )

        # Logging the information of the training dataset loaded for the current process
        if self.accelerator.is_main_process:
            self.wandb(step=step, log_dict={"training_subset": data_path_list})

        return training_dataset

    def get_test_dataset(self) -> LiteSpecFormerTestingDataset:
        """Load the dataset for model evaluation."""

        self.accelerator.print(
            f"Loading testing dataset from {self.test_data_path}", end=" -> "
        )

        # Load the testing dataset from disk
        test_dataset = HFDataset.load_from_disk(
            dataset_path=self.test_data_path,
            keep_in_memory=True,
        ).select_columns(["target"])
        test_dataset.set_format("torch")

        # Building a dataset for testing the performance of the pre-trained model under different context lengths
        testing_dataset = LiteSpecFormerTestingDataset(
            tasks=test_dataset,
            prediction_length=self.forecasting_config.output_patch_size,
            test_context_lengths=self.eval_context_lengths,
        )
        self.accelerator.print(Fore.GREEN + "Done!" + Style.RESET_ALL)

        return testing_dataset

    def get_data_loader(
        self,
        dataset: Union[
            Dataset,
            IterableDataset,
            LiteSpecFormerDataset,
            LiteSpecFormerTestingDataset,
        ],
        batch_size: int,
    ) -> DataLoader:
        """
        Create a DataLoader for the given dataset.

        Parameters
        ----------
        dataset
            The dataset for which to create the DataLoader,
            the dataset can be a standard PyTorch Dataset or an IterableDataset, or a custom dataset for time series forecasting.
        batch_size
            The batch size for the DataLoader.

        Returns
        -------
        data_loader
            A DataLoader for the given dataset with the specified batch size.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            pin_memory=True,
        )

    def train_step(
        self, batch: Dict[str, Union[torch.Tensor, torch.FloatTensor, np.ndarray]]
    ) -> float:
        """
        Training one round of the `LiteSpecFormer` model.

        Parameters
        ----------
        batch
            A batch of data for model training, which includes the input context and future labels.

        Returns
        -------
        loss
            The training loss for this round of model training.
        """

        # Set the model to training mode and clear the gradients of the optimizer.
        self.model.train()
        self.optimizer.zero_grad()

        # Transform the input into a channel-independent form.
        context = batch["context"].squeeze(dim=0)
        future_target = batch["future_target"].squeeze(dim=0)

        # The feedforward of the model to obtain the prediction results and the loss for this batch
        outputs = self.model(
            context=context,
            future_target=future_target,
        )
        # Get the loss for this batch of outputs
        loss = outputs.loss

        # Waiting for all processes to complete the forward pass and obtain the loss for this batch.
        self.wait_for_everyone()

        # Backpropagation of the loss and update the model parameters through the optimizer.
        self.accelerator.backward(loss)

        # Waiting for all processes to complete the backpropagation before updating the model parameters.
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def eval_step(self, step: int, data_loader: DataLoader) -> float:
        """
        Evaluate the performance of the model on the test dataset.

        Parameters
        ----------
        step
            The current training step, which can be used for logging and visualization during evaluation.
        data_loader
            The DataLoader for the test dataset, which provides batches of data for model evaluation.

        Returns
        -------
        eval_loss
            The average evaluation loss of the model on the test dataset.
        """

        if self.accelerator.is_main_process:
            # Create a progress bar for displaying test progress.
            pbar = tqdm(total=len(data_loader))

        # Set the model to evaluation mode to disable dropout and other training-specific layers,
        # and ensure that the model's parameters are not updated during evaluation.
        self.model.eval()

        # Initialize the total loss and the number of batches for calculating the average evaluation loss.
        total_loss = torch.zeros(1, device=self.accelerator.device)
        num_batches = 0

        # Record the model's prediction results for easy visualization later
        contexts, future_targets, predictions = [], [], []

        with torch.no_grad():
            for batch in data_loader:
                # Get the context of the input and future labels.
                context = batch["context"]
                future_target = batch["future_target"]

                # Convert it into channel-independent input
                batch_size, num_channels, context_length = context.shape
                context = context.reshape(batch_size * num_channels, context_length)
                future_target = future_target.reshape(batch_size * num_channels, -1)

                # The feedforward of the model to obtain the prediction results and the loss for this batch
                outputs = self.model(
                    context=context,
                    future_target=future_target,
                )

                # The loss for this batch of outputs
                loss = outputs.loss
                total_loss += loss.item()
                num_batches += 1

                # Record the context, future goals, and prediction results.
                contexts.append(context)
                future_targets.append(future_target)
                predictions.append(outputs.quantile_preds)

                # Waiting for all processes to complete this round of evaluation.
                self.wait_for_everyone()

                # The verification loss of all processes
                gather_loss = self.gather(total_loss).mean()

                if self.accelerator.is_main_process:
                    # The main process is updating the progress bar.
                    pbar.update(1)

                    # Update progress bar information
                    pbar.set_description(
                        f"[Testing  Step {step+1}: Loss: {gather_loss.item() / num_batches:.4f}]"
                    )

        # Visualize the model prediction results after the evaluation.
        if self.accelerator.is_main_process:
            self.visualize(
                step=step,
                process_index=self.accelerator.process_index,
                context=torch.cat(contexts, dim=0),
                future_target=torch.cat(future_targets, dim=0),
                prediction=torch.cat(predictions, dim=0),
            )

        self.wait_for_everyone()

        return gather_loss.mean() / num_batches

    def save_checkpoint(self, step: int, train_loss: float, test_loss: float) -> None:
        """
        Save the checkpoint for model training.

        Parameters
        ----------
        step
            The current training step, which is used to name the checkpoint file.
        train_loss
            The training loss at the current training step, which is used to name the checkpoint file.
        test_loss
            The testing loss at the current training step, which is used to name the checkpoint file.

        Returns
        -------
            None
        """

        # Create and save the address
        save_path = os.path.join(
            self.checkpoint_dir, f"checkpoint-{step}-{train_loss:.5f}-{test_loss:.5f}"
        )
        self.accelerator.print(f"🤖 Saving checkpoint to {save_path}", end=" -> ")

        # Save model breakpoint file
        self.accelerator.save_state(output_dir=save_path, safe_serialization=True)

        self.accelerator.print(Fore.GREEN + "Done" + Style.RESET_ALL)

    def load_checkpoint(self, pronounce: Optional[str] = True) -> None:
        """
        Load the checkpoint for model training.

        Parameters
        ----------
        pronounce
            Whether to print the information of loading checkpoints, which is set to True by default.

        Returns
        -------
            None
        """

        if self.resume_from_checkpoint is not None:
            # Checking whether to load the checkpoints
            if pronounce:
                self.accelerator.print(
                    Fore.BLUE + "Now is loading the checkpoints" + Style.RESET_ALL,
                    end=" -> ",
                )

            if self.model_params_only:
                # Load only the model parameters from the checkpoint file
                paramters = load_file(
                    path.join(self.resume_from_checkpoint, "model.safetensors")
                )
                self.model.load_state_dict(paramters, strict=False)

            else:
                # Load the checkpoint file through accelerator
                self.accelerator.load_state(self.resume_from_checkpoint)
                self.accelerator.print(Fore.GREEN + "Done" + Style.RESET_ALL)

            # Wait for all processes to load the checkpoint before proceeding with training
            self.wait_for_everyone()

            if pronounce:
                self.accelerator.print(
                    Fore.GREEN + "Done" + Style.RESET_ALL,
                    end=" -> ",
                )

    def wandb(self, step: int, log_dict: Dict) -> None:
        """Logs the loss and learning rate to Weights & Biases"""
        wandb.log(log_dict, step=step)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes."""
        return self.accelerator.gather(tensor)

    def wait_for_everyone(self) -> None:
        """Wait for all processes to reach this point."""
        self.accelerator.wait_for_everyone()

    def print(self, message: str = None, end: str = "\n") -> None:
        self.accelerator.print(message, end=end)

    def visualize(
        self,
        step: int,
        process_index: int,
        context: torch.Tensor,
        future_target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> None:
        """
        Visualize model predictions from an evaluation pass and save them as PDF figures.

        Called at the end of :meth:`eval_step` on the main process. For each randomly
        sampled series, plots the historical context, ground-truth future values, and
        quantile forecasts via :func:`~litespecformer.utils.plot_confidence_prediction`.
        Figures are written under ``{checkpoint_dir}/plotting-{step}/``.

        The context window is truncated to the shortest length in
        ``eval_context_lengths`` so that plots are comparable across evaluation settings.

        Parameters
        ----------
        step
            Current training step, used to name the output subdirectory.
        process_index
            Index of the Accelerate process that produced the predictions, included
            in each saved filename.
        context
            Input history tensor with shape ``(batch_size, context_length)``.
        future_target
            Ground-truth future values with shape ``(batch_size, prediction_length)``.
        prediction
            Quantile predictions with shape
            ``(batch_size, num_quantiles, prediction_length)``.

        Returns
        -------
            None
        """

        self.accelerator.print(f"Visualizing at step {step}...", end=" -> ")

        # Extract the shortest context distance from the context.
        context = context[:, -min(self.eval_context_lengths) :]

        # The input context, future labels, and
        # prediction results are converted into NumPy arrays for easy visualization.
        context = context.cpu().numpy()
        future_target = future_target.cpu().numpy()
        prediction = prediction.cpu().numpy()

        # Create a directory to save the visualization results.
        save_path = path.join(self.checkpoint_dir, f"plotting-{step}")
        os.makedirs(save_path, exist_ok=True)

        batch_size = context.shape[0]

        # Randomly filter the visualized index
        indices = np.random.choice(batch_size, size=128, replace=False)

        for index in indices:
            fig = plot_confidence_prediction(
                context=context[index],
                future_target=future_target[index],
                prediction=prediction[index],
            )

            # Save and close the figure to avoid memory leak
            fig.savefig(
                path.join(save_path, f"process_{process_index}_{index}.pdf"),
                bbox_inches="tight",
                dpi=128,
            )
            plt.close(fig)

    def run(
        self,
    ) -> None:
        """
        The main program for pre-training the `LiteSpecFormer` model,
        which includes the complete process of model pre-training, evaluation, checkpoint saving, and visualization.
        """
        self.wait_for_everyone()

        # Checking whether to load the checkpoints
        self.load_checkpoint(pronounce=True)

        # Get the training and testing datasets for model pre-training and evaluation
        train_dataset, test_dataset = (
            self.get_train_dataset(step=0),
            self.get_test_dataset(),
        )
        self.accelerator.wait_for_everyone()

        # Create the DataLoader for the training and testing datasets,
        # where the batch size for the training dataset is set to 1 to ensure that each process only processes one batch of data at a time,
        # while the batch size for the testing dataset is set according to the configuration.
        train_loader = self.get_data_loader(train_dataset, batch_size=1)
        test_loader = self.get_data_loader(
            test_dataset, batch_size=self.training_config.test_batch_size
        )

        # Prepare the model, optimizer, scheduler, and data loaders for distributed training through the accelerator,
        self.accelerator.print(
            "Preparing the model, optimizer, scheduler, and data loaders...", end=" -> "
        )
        (
            self.model,
            self.optimizer,
            self.scheduler,
            train_loader,
            test_loader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, train_loader, test_loader
        )
        self.wait_for_everyone()
        self.accelerator.print(Fore.GREEN + "Done" + Style.RESET_ALL)

        if self.accelerator.is_main_process:
            # Create a list to record training and validation losses.
            train_loss_list, test_loss_list = [], []

            # Create a progress bar to display the training process.
            pbar = tqdm(total=self.num_test_steps)

        else:
            # Non-main processes do not need to record losses or display progress bars.
            train_loss_list, test_loss_list = None, None

        # Start training the model
        for idx, batch in enumerate(train_loader, 1):
            # Perform one round of model training and obtain the training loss for this round,
            train_loss = self.train_step(batch)
            train_loss = self.gather(train_loss).mean()
            self.wait_for_everyone()

            # Only the main process is responsible for recording the training loss and updating the progress bar,
            # while other processes are waiting for the main process to complete these tasks before proceeding with training.
            if self.accelerator.is_main_process:
                # All the losses here need to be aggregated.
                train_loss_list.append(train_loss.item())
                pbar.update(1)
                pbar.set_description(
                    f"[Training Step {idx+1}: Loss: {train_loss_list[-1]:.4f}]"
                )

                # Log the training loss and learning rate to Weights & Biases for visualization and analysis.
                self.wandb(
                    step=idx,
                    log_dict={
                        "train_loss": train_loss_list[-1],
                        "learning_rate": self.learning_rate,
                    },
                )

            # Determine whether model evaluation is necessary.
            if idx % self.num_test_steps == 0:
                # Waiting for all processes to complete the training of this round before starting model evaluation,
                test_loss = self.eval_step(step=idx, data_loader=test_loader)
                # self.wait_for_everyone()
                # test_loss = self.gather(test_loss).mean()

                if self.accelerator.is_main_process:
                    # Record the verification loss for this time.
                    test_loss_list.append(test_loss.item())

                    # Clear the information displayed on the progress bar
                    pbar = tqdm(total=self.num_test_steps)

                    # Wandb logs the validation loss for visualization and analysis.
                    self.wandb(step=idx, log_dict={"test_loss": test_loss_list[-1]})

                # Save the model's breakpoint file during each verification.
                if self.accelerator.is_main_process:
                    self.save_checkpoint(
                        idx,
                        train_loss=train_loss_list[-1],
                        test_loss=test_loss_list[-1],
                    )

            # Determine whether it is necessary to switch the training data subset.
            if idx % self.training_config.num_subset_steps == 0:
                # Ensure all processes complete the use of old data.
                self.accelerator.wait_for_everyone()

                # Clear the old training dataset and data loader
                self.wait_for_everyone()
                train_dataset = self.get_train_dataset(step=idx)
                train_loader = self.get_data_loader(train_dataset, batch_size=1)
                train_loader = self.accelerator.prepare(train_loader)
                self.wait_for_everyone()

                # Clear the cache after all processes have finished loading the new data.
                if self.accelerator.is_main_process:
                    self.accelerator.clear()

            # Determine whether the total number of training rounds has been reached to end the training process.
            if idx >= self.num_training_steps:
                wandb.finish()
                break
