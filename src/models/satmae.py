"""
Script copied from M3LEO: https://github.com/spaceml-org/M3LEO
"""
from __future__ import annotations

from datetime import datetime

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch import nn
from torchvision.transforms import RandomResizedCrop

import wandb

from .mae_utils import masked_autoencoder_satmae, utils, vision_transformer

vit_arch_mapping = {
    "vit_t_sar_4": vision_transformer.vit_t_sar_4,
    "vit_t_sar_8": vision_transformer.vit_t_sar_8,
    "vit_t_sar_16": vision_transformer.vit_t_sar_16,
    "vit_t_sar_32": vision_transformer.vit_t_sar_32,
    "vit_s_sar_4": vision_transformer.vit_s_sar_4,
    "vit_s_sar_8": vision_transformer.vit_s_sar_8,
    "vit_s_sar_16": vision_transformer.vit_s_sar_16,
    "vit_s_sar_32": vision_transformer.vit_s_sar_32,
    "vit_b_sar_4": vision_transformer.vit_b_sar_4,
    "vit_b_sar_8": vision_transformer.vit_b_sar_8,
    "vit_b_sar_16": vision_transformer.vit_b_sar_16,
    "vit_b_sar_32": vision_transformer.vit_b_sar_32,
    "vit_l_sar_16": vision_transformer.vit_l_sar_16,
    "vit_l_sar_32": vision_transformer.vit_l_sar_32,
    "vit_h_sar_14": vision_transformer.vit_h_sar_14,
}


class MaskedAutoEncoder(pl.LightningModule):
    """ SatMAE-based Masked Autoencoder Lightning module for satellite imagery pre-training. """

    def __init__(
        self,
        mask_ratio,
        vit_arch,
        log_image_samples=8,
        num_channels=11,
        out_channels=None,  # If None, same as in_chans
        augment=False,
        image_size=256,
        learning_rate=1.5e-4,
        encode_time=False,
        encode_coords=False,
        encode_sat_angle=False,
        encode_solar_angle=False,
    ):
        """ Initialize MaskedAutoEncoder.

            Parameters
            ----------
            mask_ratio : float. Fraction of tokens to mask during pre-training.
            vit_arch : str. Key identifying the ViT architecture in vit_arch_mapping.
            log_image_samples : int. Number of samples to log as images per validation epoch (optional).
            num_channels : int. Number of input channels (optional).
            out_channels : int or None. Number of predicted output channels; defaults to num_channels (optional).
            augment : bool. If True, apply RandomResizedCrop augmentation during training (optional).
            image_size : int. Spatial resolution of input images (optional).
            learning_rate : float. Learning rate for the AdamW optimizer (optional).
            encode_time : bool. If True, add temporal positional encoding (optional).
            encode_coords : bool. If True, add spatial coordinate encoding (optional).
            encode_sat_angle : bool. If True, add satellite-angle encoding (optional).
            encode_solar_angle : bool. If True, add solar-angle encoding (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_channels = num_channels  # Changed from 3 in the paper implementation
        self.out_channels = out_channels if out_channels is not None else num_channels

        self.log_image_samples = log_image_samples
        self.image_size = image_size
        self.learning_rate = learning_rate

        self.vit_arch_name = vit_arch

        self.encode_time = encode_time
        self.encode_coords = encode_coords
        self.encode_sat_angle = encode_sat_angle
        self.encode_solar_angle = encode_solar_angle

        # self.decoder_dim = 512
        self.decoder_dim = 576  # Change to the next multiple of 4 * 6
        vit = vit_arch_mapping[vit_arch](
            num_channels=self.num_channels, image_size=self.image_size
        )

        logger.info(f"Using vit arch {self.vit_arch_name}...")

        self.mask_ratio = mask_ratio
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.num_tokens = self.sequence_length - 1
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        self.backbone = masked_autoencoder_satmae.MaskedAutoEncoderBackbone.from_vit(
            vit,
            encode_time=self.encode_time,
            encode_coords=self.encode_coords,
            encode_sat_angle=self.encode_sat_angle,
            encode_solar_angle=self.encode_solar_angle,
        )
        self.hidden_dim = self.backbone.hidden_dim
        self.decoder = masked_autoencoder_satmae.MaskedAutoEncoderDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=self.decoder_dim,
            mlp_dim=self.decoder_dim * 4,
            out_dim=vit.patch_size**2
            * self.out_channels,  # Added from the paper implementation
            dropout=0,
            attention_dropout=0,
            encode_time=self.encode_time,
            encode_coords=self.encode_coords,
            encode_sat_angle=self.encode_sat_angle,
            encode_solar_angle=self.encode_solar_angle,
        )
        self.criterion = nn.MSELoss()

        self.representation_dim = vit.seq_length * vit.hidden_dim

        self.augment = augment
        if self.augment:
            self.RandomResizedCrop = RandomResizedCrop(
                size=(self.image_size, self.image_size),
                antialias=True,
            )

        # NOTE: Previously, the positional embedding was taken from torchvision.models.vision_transformer
        # where it was defined as self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        # This was a learnable parameter.
        # In the SatMAE paper, the positional embedding is static.
        if self.encode_time:
            logger.info("Using time encoding...")
        if self.encode_coords:
            logger.info("Using coordinate encoding...")
        if self.encode_sat_angle:
            logger.info("Using satellite angle encoding...")
        if self.encode_solar_angle:
            logger.info("Using solar angle encoding...")

    def on_fit_start(self):
        """ Log hyperparameters to WandB at the start of training. """
        self.logger.experiment.log({"model/vit_arch_name": self.vit_arch_name})
        self.logger.experiment.log({"model/mask_ratio": self.mask_ratio})
        super().on_fit_start()

    def forward_encoder(
        self, images, timestamps, coords, sat_angle, solar_angle, idx_keep
    ):
        """ Encode unmasked patches with optional auxiliary positional encodings.

            Parameters
            ----------
            images : torch.Tensor. Input image stack of shape [B, num_imgs, C, H, W].
            timestamps : torch.Tensor or None. Temporal encodings per image.
            coords : torch.Tensor or None. Spatial coordinate encodings per image.
            sat_angle : torch.Tensor or None. Satellite-angle encodings per image.
            solar_angle : torch.Tensor or None. Solar-angle encodings per image.
            idx_keep : torch.Tensor. Indices of tokens to keep (unmasked).

            Returns
            -------
            torch.Tensor. Encoded representation of the unmasked tokens,
            shape [B, num_kept_tokens, hidden_dim].
        """
        return self.backbone.encode(
            images, timestamps, coords, sat_angle, solar_angle, idx_keep
        )  # Note - this ONLY calls ENCODE, not FORWARD

    def forward_decoder(
        self, x_encoded, timestamps, coords, sat_angle, solar_angle, idx_keep, idx_mask
    ):
        """ Decode encoded tokens and predict pixel values for masked patches.

            Parameters
            ----------
            x_encoded : torch.Tensor. Encoded unmasked tokens of shape [B, num_kept, hidden_dim].
            timestamps : torch.Tensor or None. Temporal encodings per image.
            coords : torch.Tensor or None. Spatial coordinate encodings per image.
            sat_angle : torch.Tensor or None. Satellite-angle encodings per image.
            solar_angle : torch.Tensor or None. Solar-angle encodings per image.
            idx_keep : torch.Tensor. Indices of the kept (unmasked) tokens.
            idx_mask : torch.Tensor. Indices of the masked tokens to predict.

            Returns
            -------
            torch.Tensor. Predicted pixel values for masked tokens,
            shape [B, num_masked, patch_size**2 * out_channels].
        """
        batch_size = x_encoded.shape[0]
        # build decoder input
        x_decode = self.decoder.embed(x_encoded)
        # extract masked tokens
        x_masked = utils.repeat_token(
            self.mask_token,
            (batch_size, (self.sequence_length - 1) * self.num_imgs + 1),
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(
            x_masked, timestamps, coords, sat_angle, solar_angle, self.num_imgs
        )

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def get_loss(self, batch, batch_idx, train=False):
        """ Run the full MAE pipeline and compute the masked reconstruction loss.

            Parameters
            ----------
            batch : dict. Batch dictionary containing 'data' and optional auxiliary tensors.
            batch_idx : int. Index of the current batch.
            train : bool. If True, apply augmentation (when self.augment=True) (optional).

            Returns
            -------
            torch.Tensor. Scalar MSE reconstruction loss over masked patches.
        """
        images = batch["data"]

        if len(images.shape) == 4:
            # Add num_imgs dimension if only one image is passed
            # If a stack of images is returned, the shape of images will already be [B, [num_imgs], C, H, W]
            images = images.unsqueeze(1)

        if self.encode_time:
            timestamps = batch["time"]
            if len(timestamps.shape) == 2:
                # Add num_imgs dimension if only one timestep is passed
                timestamps = timestamps.unsqueeze(1)
        else:
            timestamps = None

        if self.encode_coords:
            coords = batch["coords"]
            if len(coords.shape) == 4:
                # Add num_imgs dimension if only one set of coords is passed
                coords = coords.unsqueeze(1)
        else:
            coords = None

        if self.encode_sat_angle:
            sat_angle = batch["sat_angle"]
            if len(sat_angle.shape) == 4:
                # Add num_imgs dimension if only one set of coords is passed
                sat_angle = sat_angle.unsqueeze(1)
        else:
            sat_angle = None

        if self.encode_solar_angle:
            solar_angle = batch["solar_angle"]
            if len(solar_angle.shape) == 4:
                # Add num_imgs dimension if only one set of coords is passed
                solar_angle = solar_angle.unsqueeze(1)
        else:
            solar_angle = None

        if self.augment and train:
            images = self.RandomResizedCrop(images)

        assert (
            len(images.shape) == 5
        ), f"Image shape mismatch. Expected shape [B, Num_imgs, C, H, W]."

        batch_size, self.num_imgs, channels, height, width = images.shape
        total_num_tokens = (self.sequence_length - 1) * self.num_imgs + 1
        idx_keep, idx_mask = utils.random_token_mask(
            size=(
                batch_size,
                total_num_tokens,
            ),  # This masks randomly across num_imgs, i.e. the different images are not masked the same
            mask_ratio=self.mask_ratio,
            device=images.device,
        )  # Shape = [B, (1-mask_ratio)*sequence_length]

        # Run images through encoder
        x_encoded = self.forward_encoder(
            images, timestamps, coords, sat_angle, solar_angle, idx_keep
        )  # Shape = (B, (1-mask_ratio)*num_imgs*num_tokens, hidden_dim) [4, 192, 384]

        # Returns the prediction of the masked patches only
        x_pred = self.forward_decoder(
            x_encoded, timestamps, coords, sat_angle, solar_angle, idx_keep, idx_mask
        )  # Shape = (B, (1-mask_ratio)*num_imgs*num_tokens+1, out_dim = 11*patch_size**2) [4, 577, 2816]

        # get image patches for masked tokens
        # NOTE: This assumes that the first self.out_channels are the relevant channels to predict.
        patches = utils.patchify_stacked(
            images[:, :, : self.out_channels, :, :], self.patch_size
        )  # Shape = (B, num_imgs, num_patches, channels * patch_size**2) [4, 3, 256, 2816]
        # Reshape patches tp (B, num_imgs * num_patches, channels * patch_size**2) to match shape of index
        patches = patches.reshape(
            patches.shape[0], patches.shape[1] * patches.shape[2], patches.shape[3]
        )
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(
            patches, idx_mask - 1
        )  # Shape = (batch_size, (mask_ratio)*sequence_length, 32*32) [8, 38, 1024]

        loss = self.criterion(x_pred, target)

        epoch_countfrom_one = self.current_epoch + 1
        epoch_countfrom_one % 25

        return loss

    def get_representation(self, images):
        """ Encode images and return a flattened representation vector.

            Parameters
            ----------
            images : torch.Tensor. Input image tensor passed directly to forward_encoder.

            Returns
            -------
            torch.Tensor. Flattened encoded representation of shape [B, seq_length * hidden_dim].
        """
        x_encoded = self.forward_encoder(images).flatten(
            start_dim=1
        )  # (batch size, seq_length, hidden_dim)
        return x_encoded

    def training_step(self, batch, batch_idx):
        """ Compute and log the reconstruction loss for one training batch.

            Parameters
            ----------
            batch : dict. Batch dictionary with input data and optional auxiliary tensors.
            batch_idx : int. Index of the current batch.

            Returns
            -------
            torch.Tensor. Scalar training loss.
        """
        loss = self.get_loss(batch, batch_idx, train=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True)
        # self.log("epoch_logger", epoch_logger, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Compute and log the reconstruction loss for one validation batch.

            Parameters
            ----------
            batch : dict. Batch dictionary with input data and optional auxiliary tensors.
            batch_idx : int. Index of the current batch.

            Returns
            -------
            torch.Tensor. Scalar validation loss.
        """
        self.val_batch = batch  # For image logging
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """ Log masked/predicted/original image reconstructions to WandB at epoch end. """
        images = self.val_batch["data"]

        if len(images.shape) == 4:
            # Add num_imgs dimension if only one image is passed
            # If a stack of images is returned, the shape of images will already be [B, [num_imgs], C, H, W]
            images = images.unsqueeze(1)

        if self.encode_time:
            timestamps = self.val_batch["time"]
            if len(timestamps.shape) == 2:
                # Add num_imgs dimension if only one timestep is passed
                timestamps = timestamps.unsqueeze(1)
        else:
            timestamps = None

        if self.encode_coords:
            coords = self.val_batch["coords"]
            if len(coords.shape) == 4:
                # Add num_imgs dimension if only one set of coords is passed
                coords = coords.unsqueeze(1)
        else:
            coords = None

        if self.encode_sat_angle:
            sat_angle = self.val_batch["sat_angle"]
            if len(sat_angle.shape) == 4:
                # Add num_imgs dimension if only one set of coords is passed
                sat_angle = sat_angle.unsqueeze(1)
        else:
            sat_angle = None

        if self.encode_solar_angle:
            solar_angle = self.val_batch["solar_angle"]
            if len(solar_angle.shape) == 4:
                # Add num_imgs dimension if only one set of coords is passed
                solar_angle = solar_angle.unsqueeze(1)
        else:
            solar_angle = None

        batch_size = images.shape[0]
        total_num_tokens = (self.sequence_length - 1) * self.num_imgs + 1
        idx_keep, idx_mask = utils.random_token_mask(
            size=(
                batch_size,
                total_num_tokens,
            ),  # This masks randomly across num_imgs, i.e. the different images are not masked the same
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        # Encoded representation of unmasked tokens, hidden dimension
        x_encoded = self.forward_encoder(
            images, timestamps, coords, sat_angle, solar_angle, idx_keep
        )

        # Predicted pixel values for masked tokens
        x_pred_patches = self.forward_decoder(
            x_encoded, timestamps, coords, sat_angle, solar_angle, idx_keep, idx_mask
        )
        # Define zeros for masked tokens
        x_pred_mask_patches = torch.zeros(x_pred_patches.shape).to(images.device)
        # Shape (batch_size, 193, 11*256=2816) = (batch_size, len(idx_mask), bands*token_size)

        # NOTE: This assumes that the first self.out_channels are the relevant channels to predict.
        patches = utils.patchify_stacked(
            images[:, :, : self.out_channels, :, :], self.patch_size
        )
        # Reshape patches tp (B, num_imgs * num_patches, channels * patch_size**2) to match shape of index
        patches = patches.reshape(
            patches.shape[0], patches.shape[1] * patches.shape[2], patches.shape[3]
        )
        # Add predictions where masks used to be
        patches_with_pred = utils.set_at_index(
            patches, idx_mask - 1, x_pred_patches
        )  # idx_mask - 1 for cls token # Shape = (batch_size, 256, 2816)
        pred_image = utils.unpatchify_stacked(
            patches_with_pred, self.patch_size, self.num_imgs
        )

        patches_with_zeros = utils.set_at_index(
            patches, idx_mask - 1, x_pred_mask_patches
        )  # idx_mask - 1 for cls token # Shape = (batch_size, 49, 1024)
        masked_image = utils.unpatchify_stacked(
            patches_with_zeros, self.patch_size, self.num_imgs
        )

        experiment = self.logger.experiment

        if self.log_image_samples > batch_size:
            logger.warning(
                f"Reducing log_image_samples from {self.log_image_samples} to batch_size {batch_size}"
            )
            self.log_image_samples = batch_size

        if masked_image.is_cuda:
            masked_image = masked_image.cpu().numpy()
        if pred_image.is_cuda:
            pred_image = pred_image.cpu().numpy()
        if images.is_cuda:
            images = images.cpu().numpy()

        # plot to wandb

        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color="black")

        # Limit plot to 5 images to save space
        num_imgs_plot = min(self.num_imgs, 5)
        if num_imgs_plot < self.num_imgs:
            # if the sequence is longer, plot around the center image
            min_img = self.num_imgs - num_imgs_plot
            min_img + num_imgs_plot
        else:
            min_img = 0

        # plot masked input, prediction, and original image
        fig, axes = plt.subplots(
            self.log_image_samples,
            num_imgs_plot * 3,
            figsize=(num_imgs_plot * 10, self.log_image_samples * 3),
        )

        plot_channel = 5  # plots channel at index 5 (12.0um) - TODO: make this a parameter? combine multiple channels?

        for i in range(self.log_image_samples):
            for x in range(num_imgs_plot):
                if timestamps is not None:
                    date_array = timestamps[i][x + min_img][
                        :-2
                    ].int()  # remove the fractional time components
                    # Create a datetime object
                    date_time = datetime(*date_array)
                    # Format the datetime object as YYYYMMDDHHMMSS
                    time_label = date_time.strftime("(%Y-%m-%d %H:%M:%S)")
                else:
                    time_label = ""

                # set vmin and vmax based on the original image
                vmin = np.nanmin(images[i][x + min_img][plot_channel])
                vmax = np.nanmax(images[i][x + min_img][plot_channel])
                # vmin = -1
                # vmax = 1

                # plot masked input, prediction, and original image
                axes[i, x * 3].imshow(
                    np.ma.masked_equal(masked_image[i][x + min_img][plot_channel], 0),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[i, x * 3 + 1].imshow(
                    pred_image[i][x + min_img][plot_channel],
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                axes[i, x * 3 + 2].imshow(
                    images[i][x + min_img][plot_channel],
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )

                # set titles for clarity
                axes[0, x * 3].set_title("Masked")
                axes[0, x * 3 + 1].set_title("Predicted")
                axes[0, x * 3 + 2].set_title(
                    f"Original {self.val_batch['satellite'][i]} {time_label}"
                )

        fig.suptitle(f"channel {plot_channel}", y=0.92)

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

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
        """ Compute and log the reconstruction loss for one test batch.

            Parameters
            ----------
            batch : dict. Batch dictionary with input data and optional auxiliary tensors.
            batch_idx : int. Index of the current batch.

            Returns
            -------
            torch.Tensor. Scalar test loss.
        """
        loss = self.get_loss(batch, batch_idx)
        self.log("test/loss", loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        """ Configure the AdamW optimizer for training.

            Returns
            -------
            torch.optim.AdamW. Optimizer instance.
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optim

