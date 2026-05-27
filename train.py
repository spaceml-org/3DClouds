"""
Training script copied from M3LEO: https://github.com/spaceml-org/M3LEO
"""

from __future__ import annotations

import os
import sys
import warnings

from torchinfo import summary

# import pdb; pdb.set_trace()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import autoroot  # required to load from src
import dotenv
import hydra
import lightning.pytorch as pl
import omegaconf
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# MixedPrecisionPlugin
from loguru import logger
from omegaconf import DictConfig

import wandb
from src import utils
from src.models.components.checkpoint import nontrainable_breakdown

os.environ["HYDRA_FULL_ERROR"] = "1"

# load environment variables from `.env` file if it exists
dotenv.load_dotenv(override=True)

# Ignore RuntimeWarnings from angle calculations
warnings.filterwarnings("ignore", category=RuntimeWarning)

torch.set_float32_matmul_precision("medium")

@hydra.main(
    version_base="1.3",
    # config_path="configs/baselines/unet-stack",
    # config_path="configs/pre-training/multisat/sensae",
    # config_path="configs/fine-tuning/multisat/satmae-stack",
    # config_path="configs/pre-training/multisat/satmae-time-coords-angles",
    # config_path="configs/fine-tuning/multisat/satmae-sensei",
    config_path="configs/fine-tuning/multisat/satmae-time-coords-angles",
    # config_path="configs/fine-tuning/multisat/swinmae-stack",
    # config_path="configs/fine-tuning/multisat/satmae-time-coords-angles",
    config_name="train.yaml",
)
def main(config: DictConfig):
    """ Set up and run model training, then evaluate on the test set.

        Orchestrates the full training pipeline: seeding, W&B logging,
        dataloader and model instantiation (from checkpoint or scratch),
        callback/checkpoint setup, and Lightning Trainer execution.
        After training completes, loads the best checkpoint and runs
        the test loop.

        Parameters
        ----------
        config : DictConfig. Hydra configuration object containing all
                 training hyper-parameters, dataloader settings, W&B
                 credentials, and trainer options.

        Returns
        -------
        None.
    """
    # ------- seeds -------

    # extract and set model and data seeds
    seed = config.seed if "seed" in config else 42
    logger.info(f"training with seed {seed}")
    seed_everything(seed, workers=True)

    # ------- wandb logging -------

    # set up wandb config
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    # get wandb experiment tags from config
    tags = config.tags if "tags" in config else []
    if isinstance(tags, str):
        tags = tags.split()

    # get experiment name from config
    experiment_name = config.experiment_name

    # set up wandb logger
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"output dir: {output_dir}")

    wandb_logger = WandbLogger(
        name=experiment_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        tags=tags,
        save_dir=output_dir,
    )

    # log command to terminal and wandb
    cmd = " ".join(sys.argv)
    logger.info(f"Command executed: {cmd}")

    # log config to wandb
    yaml_str = omegaconf.OmegaConf.to_yaml(config)
    logger.debug(f"Hydra-config: {yaml_str}")

    # ------- dataloader -------
    logger.info("instantiating dataloader")

    dataloader = hydra.utils.instantiate(config.dataloader)

    # ------- model -------
    # load checkpoint or instantiate model from scratch
    if "load_checkpoint" in config.keys():
        hr = utils.find_hydra_run_path(
            outputs_dir=config.load_checkpoint.outputs_dir,
            wandb_runid=config.load_checkpoint.wandb_runid,
        )
        logger.info(f"hydra run path for previous model: {hr}")
        type = (
            config.load_checkpoint.type if "type" in config.load_checkpoint else "best"
        )
        model = utils.load_ckpt_from_hydra_run(hr, type=type)
    else:
        logger.info("instantiating model")
        model = hydra.utils.instantiate(config.model)

    summary(model)
    nontrainable_breakdown(model)

    torch.cuda.empty_cache()
    # ------- callbacks -------

    # Checkpoint callback
    # Define the checkpoint callback path
    dirpath = os.path.join(output_dir, "checkpoints")

    # Define the monitored metric
    monitor_metric = (
        config.get("monitor_metric") if config.get("monitor_metric") else "val/loss"
    )
    # Define the checkpoint callback
    val_checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        monitor=monitor_metric,
        save_top_k=3,
        save_last=True,
        mode="min",
        filename=f"{experiment_name}" + "-best-val-loss-epoch={epoch:02d}",
        auto_insert_metric_name=False,
    )
    callbacks = [
        val_checkpoint_callback,
    ]

    # Define whether to log additional metrics to wandb
    # Defaults to True if not included in config
    log_metrics = config.get("log_metrics") if config.get("log_metrics") else True

    # Define the training task
    # Defaults to None (i.e. not especially specified) if not included in config
    task = config.get("task") if config.get("task") else None

    # Add relevant metrics for regression tasks
    if (log_metrics) & (task == "regression"):
        # Define the checkpoint callback
        psnr_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/psnr",
            save_top_k=1,
            mode="max",
            filename=f"{experiment_name}" + "-highest-val-psnr-epoch={epoch:02d}",
            auto_insert_metric_name=False,
        )

        # Define the checkpoint callback
        ssim_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/ssim",
            save_top_k=1,
            mode="max",
            filename=f"{experiment_name}" + "-highest-val-ssim-epoch={epoch:02d}",
            auto_insert_metric_name=False,
        )
        callbacks.append(psnr_checkpoint_callback)
        callbacks.append(ssim_checkpoint_callback)

    # Add relevant metrics for segmentation tasks
    if (log_metrics) & (task == "segmentation"):
        # Define the checkpoint callback
        accuracy_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/accuracy",
            save_top_k=1,
            mode="max",
            filename=f"{experiment_name}" + "-highest-val-acc-epoch={epoch:02d}",
            auto_insert_metric_name=False,
        )

        # Define the checkpoint callback
        ssim_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/ssim",
            save_top_k=1,
            mode="max",
            filename=f"{experiment_name}" + "-highest-val-ssim-epoch={epoch:02d}",
            auto_insert_metric_name=False,
        )

        # Define the checkpoint callback
        f1_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/f1",
            save_top_k=1,
            mode="max",
            filename=f"{experiment_name}" + "-highest-val-f1-epoch={epoch:02d}",
            auto_insert_metric_name=False,
        )

        callbacks.append(accuracy_checkpoint_callback)
        callbacks.append(ssim_checkpoint_callback)
        callbacks.append(f1_checkpoint_callback)

    # ------- training details -------

    # Define plugins for trainer
    plugins = None

    # Define the precision for the model
    precision = config.get("precision") if config.get("precision") else "32-true"

    # Define whether to accumulate gradients before running optimizer
    accumulate_grad_batches = (
        config.get("accumulate_grad_batches")
        if config.get("accumulate_grad_batches")
        else 1
    )
    # Define whether to use deterministic algorithms. Defaults to False if not included in config
    if config.get("use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Define when to run validation loop
    check_val_every_n_epoch = (
        config.check_val_every_n_epoch
        if hasattr(config, "check_val_every_n_epoch")
        else 1
    )
    # Define strategy for trainer
    strategy = config.get("strategy") if config.get("strategy") else "auto"
    # Define number of nodes for distributed training
    num_nodes = config.get("num_nodes") if config.get("num_nodes") else 1
    # Define the number of devices to use
    # Uses one device if not specified in config
    devices = config.get("devices") if config.get("devices") else 1
    # Define step logging
    log_every_n_steps = (
        config.get("log_every_n_steps") if config.get("log_every_n_steps") else 50
    )
    # ------- training -------
    trainer = pl.Trainer(
        num_nodes=num_nodes,
        accelerator="gpu",
        strategy=strategy,
        devices=devices,
        plugins=plugins,
        max_epochs=config.max_epochs,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,  # Disable sanity check validation steps
        # profiler="advanced",
    )

    trainer.fit(model, dataloader)

    # ------- testing -------
    logger.info(f"---- getting best model from {output_dir}")
    best_model = utils.load_ckpt_from_hydra_run(output_dir)

    trainer.test(model=best_model, datamodule=dataloader)


if __name__ == "__main__":
    main()

