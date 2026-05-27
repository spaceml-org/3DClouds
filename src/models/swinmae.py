"""
Script modified from: https://arxiv.org/pdf/2103.14030
"""

from functools import partial

import autoroot  # needed for imports from src
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger

import wandb
from src.models.swin_utils.pos_embed import (
    get_2d_sincos_pos_embed,
    get_combined_pos_embeddings,
)
from src.models.swin_utils.utils import (
    BasicBlock,
    BasicBlockUp,
    PatchEmbedding,
    PatchExpanding,
)


class SwinMAE(pl.LightningModule):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(
        self,
        # Input/Output Configuration
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 11,
        out_chans: int = None,  # If None, same as in_chans
        # MAE-specific Parameters
        mask_ratio: float = 0.75,
        masking_window: int = 4,  # Size of the masking window in number of patches/tokens.
        decoder_embed_dim=768,  # Match encoder output dimension (no projection needed). Needs to match embed_dim*(2**(len(depths)-1))
        norm_pix_loss=False,  # Whether to normalize pixel values before computing loss
        # Swin Transformer Architecture
        depths: tuple = (
            2,
            2,
            2,
            2,
        ),  # Number of transformer blocks at each hierarchical level
        embed_dim: int = 96,  # Base embedding dimension, doubles at each hierarchical level (e.g. 96 -> 192 -> 384 -> 768)
        num_heads: tuple = (
            3,
            6,
            12,
            24,
        ),  # Number of attention heads at each hierarchical level
        attention_window: int = 8,  # Size of the attention window in number of patches/tokens.. (img_size/patch_size)/attention_window needs to be an integer.
        mlp_ratio: float = 4.0,  # MLP hidden dim = embed_dim × mlp_ratio
        # Regularization & Dropout
        drop_path_rate: float = 0.0,  # Randomly drops layers during training. NOTE: set to 0.1 in original swin_mae code
        drop_rate: float = 0.0,  # Dropout rate in MLP layers
        attn_drop_rate: float = 0.0,  # Dropout rate in attention layers
        # Architecture Components
        qkv_bias: bool = True,  # Add bias to query/key/value projections
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # Layer normalization type
        patch_norm: bool = True,  # Whether to apply normalization after patch embedding
        learning_rate: float = 1.5e-4,
        # Wandb Logging
        log_image_samples: int = 8,  # Number of samples to log for visualization
        encode_time=False,
        encode_coords=False,
        encode_sat_angle=False,
        encode_solar_angle=False,
    ):
        """ Initialize SwinMAE.

            Parameters
            ----------
            img_size : int. Input image size (height and width), assumed square (optional).
            patch_size : int. Size of each image patch in pixels (optional).
            in_chans : int. Number of input channels (optional).
            out_chans : int or None. Number of output channels; defaults to in_chans when None (optional).
            mask_ratio : float. Fraction of patches to mask during pre-training (optional).
            masking_window : int. Side length of the square window used for block masking (optional).
            decoder_embed_dim : int. Embedding dimension for the decoder; must match encoder output (optional).
            norm_pix_loss : bool. If True, normalize pixel values per patch before computing loss (optional).
            depths : tuple. Number of Swin Transformer blocks at each hierarchical level (optional).
            embed_dim : int. Base embedding dimension; doubles at each level (optional).
            num_heads : tuple. Number of attention heads at each hierarchical level (optional).
            attention_window : int. Size of the local attention window in patches (optional).
            mlp_ratio : float. Ratio of MLP hidden dimension to embedding dimension (optional).
            drop_path_rate : float. Stochastic depth drop-path rate (optional).
            drop_rate : float. Dropout rate in MLP layers (optional).
            attn_drop_rate : float. Dropout rate in attention layers (optional).
            qkv_bias : bool. If True, add learnable bias to Q/K/V projections (optional).
            norm_layer : callable. Normalization layer constructor (optional).
            patch_norm : bool. If True, apply normalization after patch embedding (optional).
            learning_rate : float. Learning rate for the AdamW optimizer (optional).
            log_image_samples : int. Number of image samples to log to WandB per epoch (optional).
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
        self.mask_ratio = mask_ratio
        self.masking_window = masking_window
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.attention_window = attention_window
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.learning_rate = learning_rate
        self.decoder_embed_dim = decoder_embed_dim  # Store for use in build methods
        self.log_image_samples = log_image_samples  # Store for logging

        self.out_chans = out_chans if out_chans is not None else in_chans

        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            in_c=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.encode_time = encode_time
        self.encode_coords = encode_coords
        self.encode_sat_angle = encode_sat_angle
        self.encode_solar_angle = encode_solar_angle

        pos_hidden_dim = self.embed_dim
        # Split the hidden dimension across the positional, time and spatial embeddings
        self.split_dim = pos_hidden_dim // 8

        if encode_time:
            pos_hidden_dim -= self.split_dim
            logger.info("Using time encoding...")
        if encode_coords:
            pos_hidden_dim -= self.split_dim
            logger.info("Using coordinate encoding...")
        if encode_sat_angle:
            logger.info("Using satellite angle encoding...")
            pos_hidden_dim -= self.split_dim
        if encode_solar_angle:
            logger.info("Using solar angle encoding...")
            pos_hidden_dim -= self.split_dim

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, pos_hidden_dim), requires_grad=False
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.layers = self.build_layers()

        self.first_patch_expanding = PatchExpanding(
            dim=self.decoder_embed_dim, norm_layer=norm_layer
        )
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(self.embed_dim)
        self.decoder_pred = nn.Linear(
            self.embed_dim, patch_size**2 * self.out_chans, bias=True
        )
        self.initialize_weights()

    def initialize_weights(self):
        """ Initialize positional embeddings, mask token, and all submodule weights. """
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """ Apply Xavier-uniform or constant initialization to Linear and LayerNorm layers.

            Parameters
            ----------
            m : nn.Module. Module to initialize.

            Returns
            -------
            None.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def on_fit_start(self):
        """ Log hyperparameters to WandB at the start of training. """
        self.logger.experiment.log({"model/mask_ratio": self.mask_ratio})
        super().on_fit_start()

    def patchify(self, imgs):
        """ Split images into non-overlapping patches.

            Parameters
            ----------
            imgs : torch.Tensor. Input images of shape (N, C, H, W).

            Returns
            -------
            torch.Tensor. Patches of shape (N, L, patch_size**2 * C).
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert (
            H % p == 0 and W % p == 0
        ), f"Image size ({H}*{W}) is not divisible by patch size ({p}*{p})"

        h = w = H // p
        x = imgs.reshape(shape=(B, C, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(B, h * w, p**2 * C)
        return x

    def unpatchify(self, x):
        """ Reconstruct images from patches.

            Parameters
            ----------
            x : torch.Tensor. Patches of shape (N, L, patch_size**2 * C).

            Returns
            -------
            torch.Tensor. Reconstructed images of shape (N, C, H, W).
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(x.shape[0], -1, h * p, h * p)
        return imgs

    def window_masking(
        self,
        x: torch.Tensor,
        r: int,
        remove: bool = False,
        mask_len_sparse: bool = False,
    ):
        """ Apply block-window masking to a sequence of patch tokens.

            Parameters
            ----------
            x : torch.Tensor. Input token sequence of shape [N, L, D].
            r : int. Side length of the masking window; each window contains r*r patches.
            remove : bool. If True, remove masked patches from the sequence and return a
                sparse_restore index for later reordering (optional).
            mask_len_sparse : bool. If True, return a sparse-length mask over windows instead
                of a full-length mask over patches (optional).

            Returns
            -------
            torch.Tensor. Masked token tensor.
            torch.Tensor. Binary mask of shape [N, L] (or [N, d^2] when mask_len_sparse=True);
                0 = kept, 1 = masked.
            torch.Tensor. Sparse restore index (only returned when remove=True).
        """
        # NOTE: Now already done before calling this function
        # x = rearrange(x, "B H W C -> B (H W) C")  # Shape [B, L=(W/p * H/p), embed_dim]
        B, L, D = x.shape
        assert int(L**0.5 / r) == L**0.5 / r
        d = int(L**0.5 // r)  # d=16, the number of windows per row/column

        # Generate random noise for indexing
        noise = torch.rand(B, d**2, device=x.device)  # Shape [B, d^2]=[B, 256]
        # Sort the noise to create a random order of patches
        sparse_shuffle = torch.argsort(noise, dim=1)
        # Restore the original order
        sparse_restore = torch.argsort(
            sparse_shuffle, dim=1
        )  # needed for restoring the original order during unpatchify
        # Keep a subset of windows based on the mask ratio
        sparse_keep = sparse_shuffle[:, : int(d**2 * (1 - self.mask_ratio))]

        # Convert 1D window index to 2D window coordinates, then to patch coordinates
        # Example:
        # sparse_keep = [5, 17, 33]  # Example indices in a window
        # For sparse_keep[0] = 5
        # Window row: 5 // 16 = 0
        # Window col: 5 % 16 = 5
        # Top-left patch: 0 * 16 * 16 + 5 * 4 = 20
        index_keep_part = (
            torch.div(sparse_keep, d, rounding_mode="floor") * d * r**2
            + sparse_keep % d * r
        )
        index_keep = index_keep_part
        # Adds all additional patches in the r*r window
        # e.g. if r=4, adds patches at [0, 1, 2, 3] in the window
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat(
                    [index_keep, index_keep_part + int(L**0.5) * i + j], dim=1
                )

        # All possible patch indices
        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)  # Shape [B, L]
        index_mask = np.zeros(
            [B, int(L - index_keep.shape[-1])], dtype=int
        )  # Shape [B, (1 - mask_ratio) * L]
        # Fill the index_mask with indices not in index_keep
        for i in range(B):
            index_mask[i] = np.setdiff1d(
                index_all[i], index_keep.cpu().numpy()[i], assume_unique=True
            )
        index_mask = torch.tensor(index_mask, device=x.device)

        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(
            index_shuffle, dim=1
        )  # needed for restoring the original order during unpatchify

        if mask_len_sparse:
            # Creates a mask for which windows are kept/masked
            mask = torch.ones([B, d**2], device=x.device)
            mask[:, : sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            # Creates a mask for which patches are kept/masked
            mask = torch.ones([B, L], device=x.device)
            mask[:, : index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)

        # TODO: Check removal of masked tokens for computational efficiency
        if remove:  # Remove the masked patches
            # If we remove the masked patches, we gather only the kept patches
            x_masked = torch.gather(
                x, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D)
            )
            x_masked = rearrange(
                x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5)
            )
            return x_masked, mask, sparse_restore
        else:  # Keep the masked patches
            # Keep the masked patches, fill them with the mask token (zero in this case)
            x_masked = torch.clone(x)
            for i in range(B):
                x_masked[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
            x_masked = rearrange(
                x_masked, "B (H W) C -> B H W C", H=int(x_masked.shape[1] ** 0.5)
            )  # Shape [B, H/p, W/p, embed_dim]
            return x_masked, mask

    def build_layers(self):
        """ Build the hierarchical Swin Transformer encoder layers.

            Returns
            -------
            nn.ModuleList. List of BasicBlock encoder layers with patch merging.
        """
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            # Creates the blocks of Swin transformers in each layer
            layer = BasicBlock(
                index=i,
                depths=self.depths,  # e.g. (2, 2, 6, 2)
                embed_dim=self.embed_dim,  # e.g. (96, 192, 384, 768)
                num_heads=self.num_heads,  # e.g. (3, 6, 12, 24)
                drop_path=self.drop_path,
                window_size=self.attention_window,  # Size of the attention window
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True,
            )
            layers.append(layer)
        return layers

    def build_layers_up(self):
        """ Build the hierarchical Swin Transformer decoder (up-sampling) layers.

            Returns
            -------
            nn.ModuleList. List of BasicBlockUp decoder layers with patch expanding.
        """
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,  # Use decoder dimensions, not encoder
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.attention_window,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
            )
            layers_up.append(layer)
        return layers_up

    def forward_encoder(
        self,
        x,
        timestamps=None,
        coords=None,
        sat_angle=None,
        solar_angle=None,
        mask_tokens=True,
    ):
        """ Encode input images with optional positional encodings and window masking.

            Parameters
            ----------
            x : torch.Tensor. Input image tensor of shape [B, C, H, W].
            timestamps : torch.Tensor or None. Temporal encoding tensor (optional).
            coords : torch.Tensor or None. Spatial coordinate encoding tensor (optional).
            sat_angle : torch.Tensor or None. Satellite-angle encoding tensor (optional).
            solar_angle : torch.Tensor or None. Solar-angle encoding tensor (optional).
            mask_tokens : bool. If True, apply window masking during pre-training (optional).

            Returns
            -------
            torch.Tensor. Encoded latent tensor of shape [B, H', W', decoder_embed_dim].
            torch.Tensor or None. Binary patch mask [B, L]; None when mask_tokens=False.
        """
        x = self.patch_embed(
            x
        )  # Shape [B, H/p, W/p, embed_dim] = [B, 256/4, 256/4, 96]

        # Add positional embeddings
        x = rearrange(x, "B H W C -> B (H W) C")  # Shape [B, L, embed_dim]
        batch_size, num_tokens, _ = x.shape
        # NOTE: Added positional encoding.
        # The SwinMAE paper did not use any positional encoding it seems.
        embed = get_combined_pos_embeddings(
            pos_embed=self.pos_embed,
            timestamps=timestamps,
            coords=coords,
            sat_angle=sat_angle,
            solar_angle=solar_angle,
            split_dim=self.split_dim,
            num_tokens=num_tokens,
            batch_size=batch_size,
        )
        x = x + embed  # Add positional embeddings
        # mask patches in x, return masked mask
        if mask_tokens:
            # During pre-taining, we want to mask patches
            x, mask = self.window_masking(
                x, r=self.masking_window, remove=False, mask_len_sparse=False
            )
        else:
            # During fine-tuning, we do not want to mask patches
            x = rearrange(x, "B (H W) C -> B H W C", H=int(x.shape[1] ** 0.5))
            mask = None
        for (
            layer
        ) in (
            self.layers
        ):  # [B, 64, 64, 96] --> [B, 32, 32, 192] --> [B, 16, 16, 384] --> [B, 8, 8, 768]
            x = layer(x)
        return x, mask

    def forward_decoder(
        self, x, timestamps=None, coords=None, sat_angle=None, solar_angle=None
    ):
        """ Decode the latent representation back to patch-pixel space.

            Parameters
            ----------
            x : torch.Tensor. Latent tensor from the encoder of shape [B, H', W', decoder_embed_dim].
            timestamps : torch.Tensor or None. Temporal encoding tensor (optional).
            coords : torch.Tensor or None. Spatial coordinate encoding tensor (optional).
            sat_angle : torch.Tensor or None. Satellite-angle encoding tensor (optional).
            solar_angle : torch.Tensor or None. Solar-angle encoding tensor (optional).

            Returns
            -------
            torch.Tensor. Predicted patches of shape [B, L, patch_size**2 * out_chans].
        """
        x = self.first_patch_expanding(
            x
        )  # [B, 8, 8, 768] --> [B, 8, 8, 1536] --> [B, 16, 16, 768]
        for layer in self.layers_up:
            x = layer(x)
        x = self.norm_up(x)  # Shape [B, H/p, W/p, embed_dim] = [B, 64, 64, 96]
        x = rearrange(
            x, "B H W C -> B (H W) C"
        )  # Shape [B, L=(H/p * W/p), embed_dim] = [B, 4096, 96]
        batch_size, num_tokens, _ = x.shape
        # TODO: Check if this is needed here
        # Add positional embeddings
        embed = get_combined_pos_embeddings(
            pos_embed=self.pos_embed,
            timestamps=timestamps,
            coords=coords,
            sat_angle=sat_angle,
            solar_angle=solar_angle,
            split_dim=self.split_dim,
            num_tokens=num_tokens,
            batch_size=batch_size,
        )
        x = x + embed  # Add positional embeddings

        x = self.decoder_pred(
            x
        )  # Shape [B, L, patch_size**2 * in_chans] = [B, 4096, 176] for 11 channel input
        return x

    def forward_loss(self, imgs, pred, mask):
        """ Compute masked reconstruction loss between predictions and target patches.

            Parameters
            ----------
            imgs : torch.Tensor. Original images of shape [N, C, H, W].
            pred : torch.Tensor. Predicted patches of shape [N, L, patch_size**2 * out_chans].
            mask : torch.Tensor. Binary mask [N, L]; 0 = keep, 1 = remove.

            Returns
            -------
            torch.Tensor. Scalar mean reconstruction loss over masked patches.
        """
        # NOTE: This assumes that the first self.out_chans are the relevant channels to predict.
        imgs = imgs[:, : self.out_chans, :, :]  # Shape [B, 11, 256, 256]
        target = self.patchify(imgs)  # Shape [B, L, patch_size**2 * out_chans]
        if self.norm_pix_loss:
            # This helps to focus on structure rather than brightness
            # e.g. clouds might be brighter (have higher pixel values) than land
            # meaning the model could focus on better predicting pixels with higher values to reduce loss
            # Patch normalization removes this bias
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        # Calculate the loss only for the masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def get_loss(self, batch, batch_idx):
        """ Run encoder–decoder pipeline and return the masked reconstruction loss.

            Parameters
            ----------
            batch : dict. Batch dictionary containing 'data' and optional auxiliary tensors.
            batch_idx : int. Index of the current batch.

            Returns
            -------
            torch.Tensor. Scalar masked reconstruction loss.
        """
        x = batch["data"]
        coords = batch["coords"] if self.encode_coords else None
        timestamps = batch["time"] if self.encode_time else None
        sat_angle = batch["sat_angle"] if self.encode_sat_angle else None
        solar_angle = batch["solar_angle"] if self.encode_solar_angle else None

        latent, mask = self.forward_encoder(
            x=x,
            coords=coords,
            timestamps=timestamps,
            solar_angle=solar_angle,
            sat_angle=sat_angle,
        )  # Shape [B, 8, 8, 768], how it comes out of the final Swin block
        pred = self.forward_decoder(
            x=latent,
            coords=coords,
            timestamps=timestamps,
            solar_angle=solar_angle,
            sat_angle=sat_angle,
        )
        loss = self.forward_loss(x, pred, mask)
        return loss

    def forward(self, batch):  # NOTE now taking a full batch and extracting data etc
        """ Run a full encoder–decoder forward pass and return loss, predictions, and mask.

            Parameters
            ----------
            batch : dict. Batch dictionary containing 'data' and optional auxiliary tensors.

            Returns
            -------
            tuple. (loss, pred, mask) where loss is a scalar tensor, pred is the predicted
            patch tensor [B, L, patch_size**2 * out_chans], and mask is the binary mask [B, L].
        """
        x = batch["data"]
        coords = batch["coords"] if self.encode_coords else None
        timestamps = batch["time"] if self.encode_time else None
        sat_angle = batch["sat_angle"] if self.encode_sat_angle else None
        solar_angle = batch["solar_angle"] if self.encode_solar_angle else None

        latent, mask = self.forward_encoder(
            x=x,
            coords=coords,
            timestamps=timestamps,
            solar_angle=solar_angle,
            sat_angle=sat_angle,
        )  # Shape [B, 8, 8, 768], how it comes out of the final Swin block
        pred = self.forward_decoder(
            x=latent,
            coords=coords,
            timestamps=timestamps,
            solar_angle=solar_angle,
            sat_angle=sat_angle,
        )
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        """ Compute and log the masked reconstruction loss for one training batch.

            Parameters
            ----------
            batch : dict. Batch dictionary with input data and optional auxiliary tensors.
            batch_idx : int. Index of the current batch.

            Returns
            -------
            torch.Tensor. Scalar training loss.
        """
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """ Compute and log the masked reconstruction loss for one validation batch.

            Parameters
            ----------
            batch : dict. Batch dictionary with input data and optional auxiliary tensors.
            batch_idx : int. Index of the current batch.

            Returns
            -------
            torch.Tensor. Scalar validation loss.
        """
        self.val_batch = batch
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """ Compute and log the masked reconstruction loss for one test batch.

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

    def on_validation_epoch_end(self):
        """ Log masked/predicted/original image reconstructions to WandB at epoch end. """
        images = self.val_batch["data"]
        coords = self.val_batch["coords"] if self.encode_coords else None
        timestamps = self.val_batch["time"] if self.encode_time else None
        sat_angle = self.val_batch["sat_angle"] if self.encode_sat_angle else None
        solar_angle = self.val_batch["solar_angle"] if self.encode_solar_angle else None

        batch_size = images.shape[0]

        # Get the latent representation and mask
        latent, mask = self.forward_encoder(
            x=images,
            coords=coords,
            timestamps=timestamps,
            solar_angle=solar_angle,
            sat_angle=sat_angle,
        )  # Shape [B, 8, 8, 768], mask is binary [B, L] where 0=keep, 1=mask
        # Run through the decoder
        pred_patches = self.forward_decoder(
            x=latent,
            coords=coords,
            timestamps=timestamps,
            solar_angle=solar_angle,
            sat_angle=sat_angle,
        )
        # Unpatchify the predicted patches to get the full reconstruction
        # TODO: Check visualization of predictions. At the moment, this predicts both masked and unmasked tokens.
        pred_images = self.unpatchify(
            pred_patches
        )  # Shape [B, 3, 256, 256] - Full reconstruction

        # Extract all patches from the original image
        # NOTE: This assumes that the first self.out_chans are the relevant channels to visualize.
        patches = self.patchify(
            images[:, : self.out_chans, :, :]
        )  # Shape [B, 4096, 176]

        # Create masked version by zeroing out masked patches for visualization
        patches_with_zeros = patches.clone()
        mask_expanded = mask.unsqueeze(-1).expand_as(patches)  # Shape [B, L, patch_dim]
        patches_with_zeros = patches_with_zeros * (
            1 - mask_expanded
        )  # Keep unmasked, zero masked

        # Unpatchify to get masked image for visualization
        masked_images = self.unpatchify(patches_with_zeros)  # Shape [B, C, H, W]

        # Set experiment
        experiment = self.logger.experiment

        if self.log_image_samples > batch_size:
            logger.warning(
                f"Reducing log_image_samples from {self.log_image_samples} to batch_size {batch_size}"
            )
            self.log_image_samples = batch_size

        # Convert to numpy for plotting
        if masked_images.is_cuda:
            masked_images = masked_images.cpu().numpy()
        if pred_images.is_cuda:
            pred_images = pred_images.cpu().numpy()
        if images.is_cuda:
            images = images.cpu().numpy()

        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color="black")

        # plot masked input, prediction, and original image
        fig, axes = plt.subplots(
            self.log_image_samples, 3, figsize=(10, self.log_image_samples * 3)
        )

        plot_channel = self.out_chans - 1

        for i in range(self.log_image_samples):
            # set vmin and vmax based on the original image
            vmin = np.nanmin(images[i][plot_channel])
            vmax = np.nanmax(images[i][plot_channel])

            # plot masked input, prediction, and original image
            axes[i, 0].imshow(
                np.ma.masked_equal(masked_images[i][plot_channel], 0),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[i, 1].imshow(
                pred_images[i][plot_channel], vmin=vmin, vmax=vmax, cmap=cmap
            )
            axes[i, 2].imshow(images[i][plot_channel], vmin=vmin, vmax=vmax, cmap=cmap)

        # set titles for clarity
        axes[0, 0].set_title("Masked")
        axes[0, 1].set_title("Predicted")
        axes[0, 2].set_title(f"Original {self.val_batch['satellite'][i]}")
        fig.suptitle(f"channel {plot_channel}", y=0.92)

        # upload to wandb
        experiment.log(
            {f"images/{self.current_epoch}_og_masked_pred": wandb.Image(fig)}
        )
        plt.close(fig)

    def configure_optimizers(self):
        """ Configure the AdamW optimizer for training.

            Returns
            -------
            torch.optim.AdamW. Optimizer instance.
        """
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optim

