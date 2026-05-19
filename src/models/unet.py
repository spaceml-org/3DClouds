r# This script contains the class defining the
# neural network used in this study. This includes
# a definition of the initial setup, forward pass,
# loss and error metrics that are conducted for every
# step of the training and validation routine.

from __future__ import annotations

import autoroot  # required for import from src
import lightning.pytorch as pl
import numpy as np  # NOTE: Added for the debugging
import torch
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import pytorch_lightning as pl
from src.models.losses import MSELoss
from src.models.sensae import SenseiAutoencoder
from src.models.unet_utils import resnet_3d, utils

# Import modules
from src.validation_utils import log_metrics_wandb, plot_multi_profiles


class BruningUNet(pl.LightningModule):
    def __init__(
        self,
        input_dim=[11, 256, 256],
        dropout=0.05,
        start_filters=32,
        height_bins=15,
        depth=4,
        lr=0.001,
        min_cs=-30.0,  # NOTE: Miminum reflectivity for un-normalization
        max_cs=20.0,  # NOTE: Maximum reflectivity for un-normalization
        mode="ResU",
        weight_decay=0.0001,
        random_plots=False,
        num_val_batches=1,
        variable="Radar_Reflectivity",
        loss_scaling: list | None = None,
        loss=None,
        sensei: bool | SenseiAutoencoder | None = None,
        sensei_embedding_size: int | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define input params
        self.input_dims = input_dim
        self.dropout = dropout
        self.start_filters = start_filters
        self.height_bins = height_bins
        self.depth = depth
        self.learning_rate = lr
        self.min_cs = min_cs
        self.max_cs = max_cs
        self.mode = mode
        self.weight_decay = weight_decay
        self.variable = [variable] if isinstance(variable, str) else variable
        self.n_variable = len(self.variable)
        self.loss_scaling = (
            torch.asarray([1] * self.n_variable).to("cuda", dtype=torch.float32)
            if loss_scaling is None
            else torch.asarray(loss_scaling).to("cuda", dtype=torch.float32)
        )

        # TODO: Make this more modular
        self.loss = MSELoss() if loss is None else loss
        # self.loss = SSIMLoss()

        self.plotting_batch = None
        self.batch_idx = None
        self.random_plots = random_plots
        self.num_val_batches = num_val_batches

        if sensei is not None and sensei is not False:
            if isinstance(sensei, pl.LightningModule):
                logger.info("Using provided SenseiAutoencoder instance.")
                if hasattr(sensei, "model"):
                    self.sensei = sensei.model
                else:
                    self.sensei = sensei
            elif sensei == True:
                logger.info("Instantiating new SenseiAutoencoder instance.")
                if sensei_embedding_size is None:
                    raise ValueError(
                        "If sensei is set to True, sensei_embedding_size must be provided."
                    )
                self.sensei = SenseiAutoencoder(embedding_size=sensei_embedding_size)
            else:
                raise ValueError(
                    "sensei parameter must be a SenseiAutoencoder instance or True to use the default SenseiAutoencoder"
                )

            self.forward = self.forward_sensei

        if self.mode.lower() == "resu":
            self.net = utils.ResUnet(
                depth=self.depth,
                dropout=self.dropout,
                start_filters=self.start_filters,
                end_filters=self.n_variable * self.height_bins,
                input_dims=self.input_dims,
            )
        elif self.mode.lower() == "resnet2d":
            self.net = resnet_3d.ResUNet2D(
                input_channels=self.input_dims[0],
                output_channels=self.n_variable * self.height_bins,
                initial_layer_channels=self.start_filters,
                depth=self.depth - 1,
                dropout=self.dropout,
            )
        elif self.mode.lower() == "resnet3d":
            self.net = resnet_3d.ResUNet2DTo3D(
                input_channels=self.input_dims[0],
                output_channels=self.n_variable,
                output_height=self.height_bins,
                initial_layer_channels=self.start_filters,
                depth=self.depth - 1,
                dropout=self.dropout,
            )
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. Choose from 'ResU', 'ResNet2D', or 'ResNet3D'."
            )

    def forward_sensei(self, batch):
        return self.net(
            self.sensei.forward_encode(batch["data"], batch["sensei_encoding"])[0]
        )

    def forward(self, batch):
        out = self.net(batch["data"])
        if self.mode.lower() in ["resu", "resnet2d"]:
            # out: [B, n_variable * height_bins, H, W]
            B, C, H, W = out.shape
            if self.n_variable > 1:
                out = out.view(B, self.n_variable, self.height_bins, H, W)
            else:
                out = out.view(B, self.height_bins, H, W)
        # For resnet3d, output is already [B, n_variable, height_bins, H, W]
        return out

    def training_step(self, batch, batch_idx):
        losses = []
        metrics = []
        cs_p = self.forward(batch)
        overpass_mask = batch["overpass_mask"]

        for i, variable in enumerate(self.variable):
            cs = batch["cloudsat"][variable].transpose(-1, -2).to(dtype=cs_p.dtype)
            pred = cs_p[:, i, ...] if self.n_variable > 1 else cs_p
            loss = self.loss(cs, pred, overpass_mask)
            losses.append(loss)

            (
                mse,
                rmse,
                ssim,
                psnr,
                mssim,
                mpsl,
                dice,
                bce,
                cloud_mse,
                clear_mse,
            ) = log_metrics_wandb(
                cs=cs,
                cs_p=pred,
                overpass_mask=overpass_mask,
                stage="train",
                experiment=self.logger.experiment,
            )
            metrics.append(
                (mse, rmse, ssim, psnr, mssim, mpsl, dice, bce, cloud_mse, clear_mse)
            )
            self.log(
                f"train/{variable}/mse", mse, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"train/{variable}/rmse",
                rmse,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"train/{variable}/ssim",
                ssim,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"train/{variable}/psnr",
                psnr,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"train/{variable}/masked_ssim",
                mssim,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"train/{variable}/masked_power_spectrum",
                mpsl,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"train/{variable}/dice",
                dice,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"train/{variable}/masked_mse_clouds",
                cloud_mse,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"train/{variable}/masked_mse_clearsky",
                clear_mse,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        mse, rmse, ssim, psnr, mssim, mpsl, dice, bce, cloud_mse, clear_mse = (
            torch.sum(torch.stack(metric) * self.loss_scaling)
            / torch.sum(self.loss_scaling)
            for metric in zip(*metrics)
        )

        self.log(
            f"train/mse",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"train/rmse",
            rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"train/ssim",
            ssim,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"train/psnr",
            psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"train/masked_ssim",
            mssim,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # self.log(
        #    f"{stage}/power_spectrum",
        #    psl,
        #    on_step=False,
        #    on_epoch=True,
        #    prog_bar=False,
        #    logger=True,
        # )
        self.log(
            f"train/masked_power_spectrum",
            mpsl,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"train/dice",
            dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # self.log(
        #    f"{stage}/bce",
        #    bce,
        #    on_step=False,
        #    on_epoch=True,
        #    prog_bar=False,
        #    logger=True,
        # )
        self.log(
            f"train/masked_mse_clouds",
            cloud_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"train/masked_mse_clearsky",
            clear_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return torch.sum(torch.stack(losses) * self.loss_scaling)

    def validation_step(self, batch, batch_idx):
        if not self.random_plots:
            self.batch_idx = batch_idx
            self.plotting_batch = batch
        if self.batch_idx is None:
            self.batch_idx = np.random.choice(self.num_val_batches)
        if (self.plotting_batch is None) & (batch_idx == self.batch_idx):
            self.plotting_batch = batch

        losses = []
        metrics = []
        cs_p = self.forward(batch)
        overpass_mask = batch["overpass_mask"]

        for i, variable in enumerate(self.variable):
            cs = batch["cloudsat"][variable].transpose(-1, -2).to(dtype=cs_p.dtype)
            pred = cs_p[:, i, ...] if self.n_variable > 1 else cs_p
            loss = self.loss(cs, pred, overpass_mask)
            losses.append(loss)
            (
                mse,
                rmse,
                ssim,
                psnr,
                mssim,
                mpsl,
                dice,
                bce,
                cloud_mse,
                clear_mse,
            ) = log_metrics_wandb(
                cs=cs,
                cs_p=pred,
                overpass_mask=overpass_mask,
                stage="val",
                experiment=self.logger.experiment,
            )
            metrics.append(
                (mse, rmse, ssim, psnr, mssim, mpsl, dice, bce, cloud_mse, clear_mse)
            )
            # Log per-variable metrics
            self.log(
                f"val/{variable}/mse", mse, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"val/{variable}/rmse", rmse, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"val/{variable}/ssim", ssim, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"val/{variable}/psnr", psnr, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"val/{variable}/masked_ssim",
                mssim,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"val/{variable}/masked_power_spectrum",
                mpsl,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"val/{variable}/dice", dice, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"val/{variable}/masked_mse_clouds",
                cloud_mse,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"val/{variable}/masked_mse_clearsky",
                clear_mse,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        mse, rmse, ssim, psnr, mssim, mpsl, dice, bce, cloud_mse, clear_mse = (
            torch.sum(torch.stack(metric) * self.loss_scaling)
            / torch.sum(self.loss_scaling)
            for metric in zip(*metrics)
        )

        self.log(
            f"val/mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"val/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"val/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"val/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"val/masked_ssim",
            mssim,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"val/masked_power_spectrum",
            mpsl,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"val/dice", dice, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            f"val/masked_mse_clouds",
            cloud_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"val/masked_mse_clearsky",
            clear_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        total_loss = torch.sum(torch.stack(losses) * self.loss_scaling)
        self.log(
            "val/loss", total_loss.item(), on_epoch=True, on_step=True, sync_dist=True
        )
        return total_loss

    def test_step(self, batch, batch_idx):
        losses = []
        metrics = []
        cs_p = self.forward(batch)
        overpass_mask = batch["overpass_mask"]

        for i, variable in enumerate(self.variable):
            cs = batch["cloudsat"][variable].transpose(-1, -2).to(dtype=cs_p.dtype)
            pred = cs_p[:, i, ...] if self.n_variable > 1 else cs_p
            loss = self.loss(cs, pred, overpass_mask)
            losses.append(loss)
            (
                mse,
                rmse,
                ssim,
                psnr,
                mssim,
                mpsl,
                dice,
                bce,
                cloud_mse,
                clear_mse,
            ) = log_metrics_wandb(
                cs=cs,
                cs_p=pred,
                overpass_mask=overpass_mask,
                stage="test",
                experiment=self.logger.experiment,
            )
            metrics.append(
                (mse, rmse, ssim, psnr, mssim, mpsl, dice, bce, cloud_mse, clear_mse)
            )
            self.log(
                f"test/{variable}/mse", mse, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"test/{variable}/rmse", rmse, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"test/{variable}/ssim", ssim, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"test/{variable}/psnr", psnr, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"test/{variable}/masked_ssim",
                mssim,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"test/{variable}/masked_power_spectrum",
                mpsl,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"test/{variable}/dice", dice, on_step=False, on_epoch=True, logger=True
            )
            self.log(
                f"test/{variable}/masked_mse_clouds",
                cloud_mse,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"test/{variable}/masked_mse_clearsky",
                clear_mse,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        mse, rmse, ssim, psnr, mssim, mpsl, dice, bce, cloud_mse, clear_mse = (
            torch.sum(torch.stack(metric) * self.loss_scaling)
            / torch.sum(self.loss_scaling)
            for metric in zip(*metrics)
        )

        self.log(
            f"test/mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"test/rmse", rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"test/ssim", ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"test/psnr", psnr, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"test/masked_ssim",
            mssim,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"test/masked_power_spectrum",
            mpsl,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"test/dice",
            dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"test/masked_mse_clouds",
            cloud_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"test/masked_mse_clearsky",
            clear_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        total_loss = torch.sum(torch.stack(losses) * self.loss_scaling)
        self.log("test/loss", total_loss.item(), on_epoch=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.learning_rate), weight_decay=self.weight_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "patience": 10,
                "interval": "epoch",
                "monitor": "val/loss",
                "frequency": 1,
            },
        }

    def on_validation_epoch_end(self):
        experiment = self.logger.experiment
        batch = self.plotting_batch

        cs_p = self.forward(batch)
        overpass_mask = batch["overpass_mask"]
        log_image_samples = 4

        # choose an infrared channel to plot
        plot_wavelength = 10300.0  # random ir wavelenth
        source_wavelengths = [
            float(batch["wavelengths"][i][0].to("cpu"))
            for i in range(len(batch["wavelengths"]))
        ]

        # match wavelength to band
        # find distance between self.wavelengths and source_wavelengths
        distances = np.abs(
            np.array(source_wavelengths)[:, None] - np.array(plot_wavelength)
        )
        # for each, find the index of the closest wavelength
        plot_channel = int(np.argmin(distances, axis=0))

        if len(cs_p.shape) == 4:  # unsqueeze if only one variable is predicted
            cs_p = cs_p.unsqueeze(1)

        # Plot all variables together
        plot_multi_profiles(
            x=batch["data"],
            cs=batch["cloudsat"],
            cs_p=cs_p,
            overpass_mask=overpass_mask,
            current_epoch=self.current_epoch,
            log_image_samples=log_image_samples,
            experiment=experiment,
            plot_channel=plot_channel,
            satellite=batch["satellite"],
            batch_idx="batch " + str(self.batch_idx),
            task="regression-"
            + ("-").join([var.lower().replace(" ", "-") for var in self.variable]),
        )

        self.plotting_batch = None
        self.batch_idx = None

