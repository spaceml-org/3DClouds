from typing import List, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from torch import nn

from src.senseiv2.embedding import *


class SenseiAutoencoder(pl.LightningModule):
    def __init__(
        self,
        embedding_size: int,
        criterion=None,
        in_features_des=0,
        log_image_samples=8,
        learning_rate=1.5e-4,
        mr=None,
        mutually_exclusive=None,
        m1_percentiles: Optional[List[float]] = [
            0.001,
            0.01,
            0.05,
            0.1,
            0.15,
            0.20,
            0.25,
            0.5,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
            0.999,
        ],
        m2_blocks: List[int] = [256, 256, 256],
        m3_d_model: int = 256,
        m3_nhead: int = 4,
        m3_dim_feedforward: int = 256,
        m3_dropout: float = 0.2,
        m3_num_layers: int = 2,
        m4_embedding_dims: int = 32,
        m4_head_layer_sizes: List[int] = [128, 32],
        m4_skips_heads: bool = False,
        m4_normalize: bool = True,
        num_head: int = 2,
    ):
        super().__init__()

        self.mr = mr
        self.mutually_exclusive = mutually_exclusive

        self.sensei_encodeA = SpectralEmbeddingSimpleA(
            in_features=embedding_size,
            m1_percentiles=m1_percentiles,
            m2_blocks=m2_blocks,
            m3_d_model=m3_d_model,
            m3_nhead=m3_nhead,
            m3_dim_feedforward=m3_dim_feedforward,
            m3_dropout=m3_dropout,
            m3_num_layers=m3_num_layers,
            m4_embedding_dims=m4_embedding_dims,
            num_head=num_head,
            device="cuda",
        )

        self.sensei_encodeB = SpectralEmbeddingSimpleB(
            m3_d_model=m3_d_model,
            m4_embedding_dims=m4_embedding_dims,
            m4_head_layer_sizes=m4_head_layer_sizes,
            m4_skips_heads=m4_skips_heads,
            m4_normalize=m4_normalize,
            num_head=num_head,
            device="cuda",
        )

        self.sensei_decode = SpectralEmbeddingReverseSimple(
            # in_features=in_features_des,
            in_features=m3_d_model,
            m1_percentiles=m1_percentiles,
            m2_blocks=m2_blocks,
            m3_d_model=m3_d_model,
            m3_nhead=m3_nhead,
            m3_dim_feedforward=m3_dim_feedforward,
            m3_dropout=m3_dropout,
            m3_num_layers=m3_num_layers,
            m4_embedding_dims=m4_embedding_dims,
            m4_head_layer_sizes=m4_head_layer_sizes,
            m4_skips_heads=m4_skips_heads,
            m4_normalize=m4_normalize,
            num_head=num_head,
            device="cuda",
        )

        self.criterion = nn.MSELoss() if criterion is None else criterion
        self.log_image_samples = log_image_samples
        self.learning_rate = learning_rate

    def addInfoToBand(self, batch_sensor_info_k, batch):
        time_dict = {"fyear": batch["time"][:, 6], "fday": batch["time"][:, 6]}
        angles_dict = {
            "center_solar_angle_alfa": batch["center_solar_angle"][:, 0],
            "center_solar_angle_beta": batch["center_solar_angle"][:, 1],
            "center_sat_angle_alfa": batch["center_sat_angle"][:, 0],
            "center_sat_angle_beta": batch["center_sat_angle"][:, 1],
            "center_lat": batch["center_coords"][:, 0],
            "center_lon": batch["center_coords"][:, 1],
        }
        return batch_sensor_info_k | time_dict | angles_dict

    def addInfoToSensorInfo(self, sensor_info, batch):
        return {
            k: self.addInfoToBand(sensor_info[k], batch) for k in sensor_info.keys()
        }

    def get_mem_idx(self, mem, idx):
        if isinstance(mem, list):
            return [self.get_mem_idx(el, idx) for el in mem]
        else:
            return mem[:, idx, :]

    def get_idxs(self, num_channels, train=False):
        if (self.mr is not None) & train & (self.mr > 0):
            num_mask_channels = int(np.floor(self.mr * num_channels))
            idx_mask = np.random.choice(num_channels, num_mask_channels, replace=False)

            if self.mutually_exclusive:
                idx_keep = np.array(
                    list(set(np.arange(num_channels)).difference(set(idx_mask)))
                )
            else:
                idx_keep = np.random.choice(
                    num_channels, num_mask_channels, replace=False
                )

            idx_mask = torch.tensor(idx_mask, dtype=torch.long, device=self.device)
            idx_keep = torch.tensor(idx_keep, dtype=torch.long, device=self.device)

        else:
            idx_mask = None
            idx_keep = None

        return idx_mask, idx_keep

    def forward_encode(self, images, encoded_descriptors, idx_keep=None, idx_mask=None):
        memory = self.sensei_encodeA(images, encoded_descriptors)
        memory_keep = memory
        memory_mask = memory
        if idx_keep is not None:
            images = images[:, idx_keep, ...]
            memory_keep = self.get_mem_idx(memory_keep, idx_keep)

        encoded_bands = self.sensei_encodeB(images, memory_keep)
        if idx_mask is not None:
            memory_mask = self.get_mem_idx(memory_mask, idx_mask)

        return encoded_bands, memory_mask

    def forward(self, batch, idx_keep=None, idx_mask=None):
        images = batch["data"]
        encoded_descriptors = batch["sensei_encoding"]

        encoded_bands, memory_mask = self.forward_encode(
            images,
            encoded_descriptors,
            idx_keep=idx_keep,
            idx_mask=idx_mask,
        )

        x_reconstructed = self.sensei_decode(encoded_bands, memory_mask)
        return x_reconstructed

    def on_fit_start(self):
        """
        Called by lightning at the beginning of training. This is the point at which the logger is available, so
        we can now log parameters to wandb.
        """
        self.logger.experiment.log({"encoding these properties": ""})

        super().on_fit_start()

    def get_loss(self, batch, batch_idx=None, train=False):
        images = batch["data"]
        num_channels = images.shape[1]
        idx_mask, idx_keep = self.get_idxs(num_channels, train)

        pred = self.forward(batch, idx_keep, idx_mask)

        if idx_mask is not None:
            images = images[:, idx_mask, ...]

        # use only idx mask for images
        loss = self.criterion(pred, images)

        return loss

    def training_step(self, batch):
        loss = self.get_loss(batch, train=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True)
        # self.log("epoch_logger", epoch_logger, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_batch = batch  # For image logging
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        images = self.val_batch["data"]
        num_channels = images.shape[1]
        idx_mask, idx_keep = self.get_idxs(num_channels, train=True)
        if idx_mask is not None:
            images = images[:, idx_mask, ...]
        pred = self.forward(self.val_batch, idx_keep, idx_mask)
        batch_size = images.shape[0]

        experiment = self.logger.experiment

        if self.log_image_samples > batch_size:
            logger.warning(
                f"Reducing log_image_samples from {self.log_image_samples} to batch_size {batch_size}"
            )
            self.log_image_samples = batch_size

        if pred.is_cuda:
            pred = pred.cpu().numpy()
        if images.is_cuda:
            images = images.cpu().numpy()
        if idx_keep is not None:
            if idx_keep.is_cuda:
                idx_keep = idx_keep.detach().cpu().numpy()
                idx_mask = idx_mask.detach().cpu().numpy()

        # plot to wandb

        cmap1 = plt.cm.Blues.copy()
        cmap1.set_bad(color="black")
        cmap2 = plt.cm.Purples.copy()
        cmap2.set_bad(color="black")

        # plot masked input, prediction, and original image
        fig, axes = plt.subplots(
            self.log_image_samples,
            4,
            figsize=(10, self.log_image_samples * 2),
        )

        if idx_keep is not None:
            non_seen_channels = list(set(idx_mask).difference(set(idx_keep)))
            if len(non_seen_channels) >= 2:
                plot_channel1, plot_channel2 = np.random.choice(
                    non_seen_channels, 2, replace=False
                )
                is_seen1 = is_seen2 = "not"
            elif len(non_seen_channels) == 1:
                plot_channel1 = np.random.choice(non_seen_channels)
                plot_channel2 = np.random.choice(idx_mask)
                is_seen1 = "not"
                is_seen2 = ""
            else:
                plot_channel1, plot_channel2 = np.random.choice(
                    idx_mask, 2, replace=False
                )
                is_seen1 = is_seen2 = ""

            # convert plot channel to images and preds
            plot_channel1 = np.where(plot_channel1 == idx_mask)
            plot_channel1 = int(plot_channel1[0])
            plot_channel2 = np.where(plot_channel2 == idx_mask)
            plot_channel2 = int(plot_channel2[0])
        else:
            plot_channel1, plot_channel2 = np.random.choice(
                num_channels, 2, replace=False
            )
            is_seen1 = is_seen2 = ""

        for i in range(self.log_image_samples):
            vmin = np.minimum(
                np.nanmin(images[i][plot_channel1]), np.nanmin(pred[i][plot_channel1])
            )
            vmax = np.maximum(
                np.nanmax(images[i][plot_channel1]), np.nanmax(pred[i][plot_channel1])
            )

            im1 = axes[i, 0].imshow(
                pred[i][plot_channel1],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap1,
            )
            axes[i, 1].imshow(
                images[i][plot_channel1],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap1,
            )
            plt.colorbar(
                im1,
                ax=axes[i, 0:2],
                label=plot_channel1,
                orientation="vertical",
                shrink=1,
            )

            vmin = np.minimum(
                np.nanmin(images[i][plot_channel2]), np.nanmin(pred[i][plot_channel2])
            )
            vmax = np.maximum(
                np.nanmax(images[i][plot_channel2]), np.nanmax(pred[i][plot_channel2])
            )

            im1 = axes[i, 2].imshow(
                pred[i][plot_channel2],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap2,
            )
            axes[i, 3].imshow(
                images[i][plot_channel2],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap2,
            )
            plt.colorbar(
                im1,
                ax=axes[i, 2:4],
                label=plot_channel2,
                orientation="vertical",
                shrink=1,
            )

            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])

        # set titles for clarity
        axes[0, 0].set_title("predicted")
        axes[0, 1].set_title("target")

        axes[0, 2].set_title("predicted")
        axes[0, 3].set_title("target")

        fig.text(
            0.28,
            0.92,
            f"{self.val_batch['satellite'][i]} channel {plot_channel1} was {is_seen1} seen",
            ha="center",
        )
        fig.text(
            0.68,
            0.92,
            f"{self.val_batch['satellite'][i]} channel {plot_channel2} was {is_seen2} seen",
            ha="center",
        )

        # Adjust spacing between subplots
        # plt.subplots_adjust(wspace=0.4, hspace=0.6)

        # upload to wandb
        experiment.log(
            {
                f"images/epoch_{self.current_epoch}_all_images_masked_pred": wandb.Image(
                    fig
                )
            }
        )
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("test/loss", loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optim
