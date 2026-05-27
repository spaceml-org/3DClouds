# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from datetime import datetime
import numpy as np
import torch
import autoroot # needed for import from src
from src.senseiv2.utils import encode_position_linear, encode_position_angle, torch_circmean, encode_position_spherical_angles

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------


def get_1d_sincos_pos_embed_torch(embed_dim, pos):
    """ Compute 1D sin-cos positional embeddings using PyTorch tensors.

        Parameters
        ----------
        embed_dim : int. Output embedding dimension for each position; must be even.
        pos : torch.Tensor. Positions to encode with shape (M,).

        Returns
        -------
        torch.Tensor. Positional embeddings with shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    # omega = array(0, 1, .... 64)
    omega /= embed_dim / 2.0
    # omega = array(0, 1/64, .... 1)
    omega = 1.0 / 10000**omega  # (D/2,)
    # omega = array(1, 1/10000, .... 1/10000^1)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


def get_1d_sincos_pos_embed(embed_dim, pos):
    """ Compute 1D sin-cos positional embeddings using NumPy arrays.

        Parameters
        ----------
        embed_dim : int. Output embedding dimension for each position; must be even.
        pos : np.ndarray. Positions to encode with shape (M,).

        Returns
        -------
        np.ndarray. Positional embeddings with shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """ Compute 2D sin-cos positional embeddings for a square grid.

        Parameters
        ----------
        embed_dim : int. Output embedding dimension; must be even.
        grid_size : int. Number of patches per side of the square grid.
        cls_token : bool. If True, prepends a zero embedding for the class token (optional).

        Returns
        -------
        np.ndarray. Positional embeddings with shape (H*W, embed_dim), or (H*W+1, embed_dim) if cls_token.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_height, cls_token=False):
    """ Compute 3D sin-cos positional embeddings for a volumetric grid.

        Parameters
        ----------
        embed_dim : int. Output embedding dimension; must be divisible by 3.
        grid_size : int. Number of patches per spatial side (height and width).
        grid_height : int. Number of patches along the depth dimension.
        cls_token : bool. If True, prepends a zero embedding for the class token (optional).

        Returns
        -------
        np.ndarray. Positional embeddings with shape (H*W*D, embed_dim), or (H*W*D+1, embed_dim) if cls_token.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_d = np.arange(grid_height, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_height])

    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed(embed_dim // 3, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 3, grid[1])  # (H*W, D/2)
    emb_d = get_1d_sincos_pos_embed(embed_dim // 3, grid[2])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w, emb_d], axis=1)  # (H*W, D)
    print("pos_embed shape", pos_embed.shape)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_fractional_time_embedding(timestamps, hidden_dim, num_tokens):
    """ Encode timestamps into 1D positional embeddings using fraction of year and fraction of day.

        Parameters
        ----------
        timestamps : torch.Tensor. Timestamp tensor with shape (batch_size, num_imgs, time_comps);
            time_comps[-2] is fraction_of_year and time_comps[-1] is fraction_of_day.
        hidden_dim : int. Dimension of the output embeddings; must be even.
        num_tokens : int. Number of patch tokens per image to broadcast the embedding across.

        Returns
        -------
        torch.Tensor. Temporal positional embeddings with shape (batch_size, num_imgs * num_tokens, hidden_dim).
    """
    # Assumes timestamps are in format [year, month, day, hour, minute, second, fraction_of_year, fraction_of_day]
    batch_size, num_imgs, time_comps = timestamps.shape

    timestamps = timestamps.reshape(-1, time_comps)  # Reshape to [B*num_imgs, time_comps]

    fraction_of_year = timestamps[:, -2]
    fraction_of_day = timestamps[:, -1]

    ts_embed = torch.cat([encode_position_angle(
                    torch.asarray(fraction_of_year),
                    encode_dim=hidden_dim // 2,
                    min=0,
                    max=1,
                ),
                encode_position_angle(
                    torch.asarray(fraction_of_day),
                    encode_dim=hidden_dim // 2,
                    min=0,
                    max=1,
                )], dim=-1
            ).float()

    # Reshape to [B, num_imgs, 1, hidden_dim]
    ts_embed = ts_embed.reshape(-1, num_imgs, ts_embed.shape[-1]).unsqueeze(2)
    # Reshape to [B, num_imgs, sequence_length - 1, hidden_dim]
    ts_embed = ts_embed.expand(-1, -1, num_tokens, -1)
    # Reshape to [B, num_imgs*(sequence_length - 1), hidden_dim]
    ts_embed = ts_embed.reshape(batch_size, -1, ts_embed.shape[-1])
    return ts_embed


def get_coords_embedding(coords, hidden_dim, num_tokens):
    """ Encode geographic coordinates into positional embeddings using spherical angles.

        Parameters
        ----------
        coords : torch.Tensor. Coordinate tensor with shape (batch_size, num_imgs, 2, h, w);
            channel 0 is latitude and channel 1 is longitude.
        hidden_dim : int. Dimension of the output embeddings.
        num_tokens : int. Number of patch tokens per image to broadcast the embedding across.

        Returns
        -------
        torch.Tensor. Geographic positional embeddings with shape (batch_size, num_imgs * num_tokens, hidden_dim).
    """
    # Assumes coords are in format [lat, lon]
    batch_size, num_imgs, coords_comp, h, w = coords.shape
    # Split hidden_dim across all time components
    coords = coords.reshape(-1, coords_comp, h, w)  # Reshape to [B*num_imgs, 2, h, w]

    coords_embed = encode_position_spherical_angles(
        torch.nanmean(torch.asarray(coords[:, 0, :, :]), dim=(-2, -1)),
        torch_circmean(
                    torch.asarray(coords[:, 1, :, :]),
                    low=0,
                    high=360,
                    dim=(-2, -1),
                ),
                encode_dim=hidden_dim,
                zenith_min=-90,
                zenith_max=90,
                azimuth_min=0,
                azimuth_max=360,
            ) # Shape = [B*num_imgs, hidden_dim]

    # Reshape to [B, num_imgs, 1, hidden_dim]
    coords_embed = coords_embed.reshape(-1, num_imgs, coords_embed.shape[-1]).unsqueeze(2)
    # Reshape to [B, num_imgs, sequence_length - 1, hidden_dim]
    coords_embed = coords_embed.expand(-1, -1, num_tokens, -1)
    # Reshape to [B, num_imgs*(sequence_length - 1), hidden_dim]
    coords_embed = coords_embed.reshape(batch_size, -1, coords_embed.shape[-1])
    return coords_embed


def get_angle_embedding(angle, hidden_dim, num_tokens):
    """ Encode view angles into positional embeddings using spherical angles.

        Parameters
        ----------
        angle : torch.Tensor. Angle tensor with shape (batch_size, num_imgs, 2, h, w);
            channel 0 is zenith angle and channel 1 is azimuth angle.
        hidden_dim : int. Dimension of the output embeddings.
        num_tokens : int. Number of patch tokens per image to broadcast the embedding across.

        Returns
        -------
        torch.Tensor. Angular positional embeddings with shape (batch_size, num_imgs * num_tokens, hidden_dim).
    """
    # Assumes angle is in format [zenith, azimuth]
    batch_size, num_imgs, angle_comp, h, w = angle.shape
    # Split hidden_dim across all time components
    angle = angle.reshape(-1, angle_comp, h, w)  # Reshape to [B*num_imgs, 2, h, w]

    angle_embed = encode_position_spherical_angles(
                    torch.nanmean(
                        torch.asarray(angle[:, 0, :, :]), dim=(-2, -1)
                    ),
                    torch_circmean(
                        torch.asarray(angle[:, 1, :, :]),
                        low=0,
                        high=360,
                        dim=(-2, -1),
                    ),
                    encode_dim=hidden_dim,
                    zenith_min=0,
                    zenith_max=180,
                    azimuth_min=0,
                    azimuth_max=360,
                ) # Shape = [B*num_imgs, hidden_dim]

    # Reshape to [B, num_imgs, 1, hidden_dim]
    angle_embed = angle_embed.reshape(-1, num_imgs, angle_embed.shape[-1]).unsqueeze(2)
    # Reshape to [B, num_imgs, sequence_length - 1, hidden_dim]
    angle_embed = angle_embed.expand(-1, -1, num_tokens, -1)
    # Reshape to [B, num_imgs*(sequence_length - 1), hidden_dim]
    angle_embed = angle_embed.reshape(batch_size, -1, angle_embed.shape[-1])
    return angle_embed

