from __future__ import annotations

import math
import warnings
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter


@torch.no_grad()
def batch_shuffle(
    batch: torch.Tensor, distributed: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly shuffles all tensors in the batch.

    Parameters
    ----------
    batch : torch.Tensor. The batch to shuffle.
    distributed : bool. If True then batches are shuffled across multiple gpus (optional).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]. A (batch, shuffle) tuple where batch is the shuffled
    version of the input batch and shuffle is an index to restore the original order.

    Examples:
        >>> # forward pass through the momentum model with batch shuffling
        >>> x1_shuffled, shuffle = batch_shuffle(x1)
        >>> f1 = moco_momentum(x1)
        >>> out0 = projection_head_momentum(f0)
        >>> out1 = batch_unshuffle(out1, shuffle)
    """
    if distributed:
        return batch_shuffle_distributed(batch)
    batch_size = batch.shape[0]
    shuffle = torch.randperm(batch_size, device=batch.device)
    return batch[shuffle], shuffle


@torch.no_grad()
def batch_unshuffle(
    batch: torch.Tensor,
    shuffle: torch.Tensor,
    distributed: bool = False,
) -> torch.Tensor:
    """Unshuffles a batch.

    Parameters
    ----------
    batch : torch.Tensor. The batch to unshuffle.
    shuffle : torch.Tensor. Index to unshuffle the batch.
    distributed : bool. If True then the batch is unshuffled across multiple gpus (optional).

    Returns
    -------
    torch.Tensor. The unshuffled batch.

    Examples:
        >>> # forward pass through the momentum model with batch shuffling
        >>> x1_shuffled, shuffle = batch_shuffle(x1)
        >>> f1 = moco_momentum(x1)
        >>> out0 = projection_head_momentum(f0)
        >>> out1 = batch_unshuffle(out1, shuffle)
    """
    if distributed:
        return batch_unshuffle_distributed(batch, shuffle)
    unshuffle = torch.argsort(shuffle)
    return batch[unshuffle]


@torch.no_grad()
def concat_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Returns concatenated instances of x gathered from all gpus.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    Parameters
    ----------
    x : torch.Tensor. Tensor to gather across all gpus.

    Returns
    -------
    torch.Tensor. Concatenated tensor from all gpus along dim 0.
    """
    output = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(output, x, async_op=False)
    output = torch.cat(output, dim=0)
    return output


@torch.no_grad()
def batch_shuffle_distributed(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shuffles batch over multiple gpus.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    Parameters
    ----------
    batch : torch.Tensor. The tensor to shuffle.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]. A (batch, shuffle) tuple where batch is the shuffled
    version of the input batch and shuffle is an index to restore the original order.
    """
    # gather from all gpus
    batch_size_this = batch.shape[0]
    batch_gather = concat_all_gather(batch)
    batch_size_all = batch_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    dist.broadcast(idx_shuffle, src=0)

    # index for restoring
    shuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = dist.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return batch_gather[idx_this], shuffle


@torch.no_grad()
def batch_unshuffle_distributed(
    batch: torch.Tensor, shuffle: torch.Tensor
) -> torch.Tensor:
    """Undo batch shuffle over multiple gpus.

    This code was taken and adapted from here:
    https://github.com/facebookresearch/moco.

    Parameters
    ----------
    batch : torch.Tensor. The tensor to unshuffle.
    shuffle : torch.Tensor. Index to restore the original tensor.

    Returns
    -------
    torch.Tensor. The unshuffled tensor.
    """
    # gather from all gpus
    batch_size_this = batch.shape[0]
    batch_gather = concat_all_gather(batch)
    batch_size_all = batch_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = dist.get_rank()
    idx_this = shuffle.view(num_gpus, -1)[gpu_idx]

    return batch_gather[idx_this]


def deactivate_requires_grad(model: nn.Module):
    """Deactivates the requires_grad flag for all parameters of a model.

    This has the same effect as permanently executing the model within a `torch.no_grad()`
    context. Use this method to disable gradient computation and therefore
    training for a model.

    Parameters
    ----------
    model : nn.Module. The model whose parameters should have requires_grad set to False.

    Returns
    -------
    None.

    Examples:
        >>> backbone = resnet18()
        >>> deactivate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = False


def activate_requires_grad(model: nn.Module):
    """Activates the requires_grad flag for all parameters of a model.

    Use this method to activate gradients for a model (e.g. after deactivating
    them using `deactivate_requires_grad(...)`).

    Parameters
    ----------
    model : nn.Module. The model whose parameters should have requires_grad set to True.

    Returns
    -------
    None.

    Examples:
        >>> backbone = resnet18()
        >>> activate_requires_grad(backbone)
    """
    for param in model.parameters():
        param.requires_grad = True


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float):
    """Updates parameters of `model_ema` with Exponential Moving Average of `model`

    Momentum encoders are a crucial component of models such as MoCo or BYOL.

    Parameters
    ----------
    model : nn.Module. The model providing the current parameter values.
    model_ema : nn.Module. The momentum model whose parameters are updated in-place.
    m : float. Momentum coefficient; typically close to 1 (e.g. 0.999).

    Returns
    -------
    None.

    Examples:
        >>> backbone = resnet18()
        >>> projection_head = MoCoProjectionHead()
        >>> backbone_momentum = copy.deepcopy(moco)
        >>> projection_head_momentum = copy.deepcopy(projection_head)
        >>>
        >>> # update momentum
        >>> update_momentum(moco, moco_momentum, m=0.999)
        >>> update_momentum(projection_head, projection_head_momentum, m=0.999)
    """
    for model_ema, model in zip(model_ema.parameters(), model.parameters()):
        model_ema.data = model_ema.data * m + model.data * (1.0 - m)


@torch.no_grad()
def normalize_weight(weight: nn.Parameter, dim: int = 1, keepdim: bool = True):
    """Normalizes the weight to unit length along the specified dimension.

    Parameters
    ----------
    weight : nn.Parameter. Weight tensor to normalize in-place.
    dim : int. Dimension along which to compute the norm (optional).
    keepdim : bool. Whether to keep the reduced dimension (optional).

    Returns
    -------
    None.
    """
    weight.div_(torch.norm(weight, dim=dim, keepdim=keepdim))


# copy paste from PyTorch master branch as it is not available in older releases
# source: https://github.com/pytorch/pytorch/blob/20ac7362009dd8e0aca6e72fc9357773136a83b8/torch/nn/init.py#L22-L54
def _no_grad_trunc_normal(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    """Initializes the input tensor with a truncated normal distribution.

    This method is based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Parameters
    ----------
    tensor : torch.Tensor. The tensor to initialize in-place.
    mean : float. Mean of the distribution.
    std : float. Standard deviation of the distribution.
    a : float. Lower truncation bound; values below are clamped.
    b : float. Upper truncation bound; values above are clamped.

    Returns
    -------
    torch.Tensor. The initialized tensor (same object as input).
    """

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def repeat_token(token: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Repeats a token size times.

    Parameters
    ----------
    token : torch.Tensor. Token tensor with shape (1, 1, dim).
    size : tuple[int, int]. (batch_size, sequence_length) target shape.

    Returns
    -------
    torch.Tensor. Tensor with shape (batch_size, sequence_length, dim) containing
    copies of the input token.
    """
    batch_size, sequence_length = size
    return token.repeat(batch_size, sequence_length, 1)


def expand_index_like(index: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Expands the index along the last dimension of the input tokens.

    Parameters
    ----------
    index : torch.Tensor. Index tensor with shape (batch_size, idx_length) where each
        entry is an index in [0, sequence_length).
    tokens : torch.Tensor. Tokens tensor with shape (batch_size, sequence_length, dim).

    Returns
    -------
    torch.Tensor. Index tensor with shape (batch_size, idx_length, dim) where the
    original indices are repeated dim times along the last dimension.
    """
    dim = tokens.shape[-1]
    index = index.unsqueeze(-1).expand(-1, -1, dim)
    return index


def get_at_index(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Selects tokens at index.

    Parameters
    ----------
    tokens : torch.Tensor. Token tensor with shape (batch_size, sequence_length, dim).
    index : torch.Tensor. Index tensor with shape (batch_size, index_length) where each
        entry is an index in [0, sequence_length).

    Returns
    -------
    torch.Tensor. Token tensor with shape (batch_size, index_length, dim) containing
    the selected tokens.
    """
    index = expand_index_like(index, tokens)
    return torch.gather(tokens, 1, index)


def set_at_index(
    tokens: torch.Tensor, index: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Copies all values into the input tensor at the given indices.

    Parameters
    ----------
    tokens : torch.Tensor. Tokens tensor with shape (batch_size, sequence_length, dim).
    index : torch.Tensor. Index tensor with shape (batch_size, index_length).
    value : torch.Tensor. Value tensor with shape (batch_size, index_length, dim).

    Returns
    -------
    torch.Tensor. Tokens tensor with shape (batch_size, sequence_length, dim) containing
    the new values.
    """
    index = expand_index_like(index, tokens)
    return torch.scatter(tokens, 1, index, value)


def mask_at_index(
    tokens: torch.Tensor, index: torch.Tensor, mask_token: torch.Tensor
) -> torch.Tensor:
    """Copies mask token into the input tensor at the given indices.

    Parameters
    ----------
    tokens : torch.Tensor. Tokens tensor with shape (batch_size, sequence_length, dim).
    index : torch.Tensor. Index tensor with shape (batch_size, index_length).
    mask_token : torch.Tensor. Mask token with shape (1, 1, dim).

    Returns
    -------
    torch.Tensor. Tokens tensor with shape (batch_size, sequence_length, dim) containing
    the new values.
    """
    mask = tokens.new_zeros(tokens.shape)
    mask = set_at_index(mask, index, 1)
    return (1 - mask) * tokens + mask * mask_token


def prepend_class_token(
    tokens: torch.Tensor, class_token: torch.Tensor
) -> torch.Tensor:
    """Prepends class token to tokens.

    Parameters
    ----------
    tokens : torch.Tensor. Tokens tensor with shape (batch_size, sequence_length, dim).
    class_token : torch.Tensor. Class token with shape (1, 1, dim).

    Returns
    -------
    torch.Tensor. Tokens tensor with the class token prepended at index 0 in every
    sequence, with shape (batch_size, sequence_length + 1, dim).
    """
    batch_size = tokens.shape[0]
    batch_class_token = class_token.expand(batch_size, -1, -1)
    # print(f"Batch size: {batch_size}")
    # print(f"Tokens Shape: {tokens.shape}")
    # print(f"Class token shape: {class_token.shape}")
    # print(f"Batch class token shape: {batch_class_token.shape}")
    return torch.cat([batch_class_token, tokens], dim=1)


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts a batch of input images into patches.

    Parameters
    ----------
    images : torch.Tensor. Images tensor with shape (batch_size, channels, height, width).
    patch_size : int. Patch size in pixels; height and width must be multiples of patch_size.

    Returns
    -------
    torch.Tensor. Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2)
    where num_patches = (height / patch_size) * (width / patch_size).
    """
    # N, C, H, W = (batch_size, channels, height, width)
    # Input shape is (batch_size, channels, height, width)
    batch_size, channels, height, width = images.shape

    # Assert that height and width are multiples of patch_size
    assert height % patch_size == 0, "Height is not a multiple of patch_size"
    assert width % patch_size == 0, "Width is not a multiple of patch_size"

    # Assert that height and width are equal
    assert height == width, "Height and width are not equal"

    # Use unfold to split the tensor into patches of size `patch_size`
    patches = images.unfold(2, patch_size, patch_size)
    patches = patches.unfold(3, patch_size, patch_size)

    # patches has shape (n, c, h, w, patch_size, patch_size)
    # We reshape and permute to get it into the shape (n, num_patches, c * patch_size * patch_size)

    # (n, c, h, w, patch_size, patch_size)
    num_patches = height * width // patch_size**2
    patches = patches.reshape(
        batch_size, channels, num_patches, patch_size * patch_size
    )  # (n, c, num_patches, patch_size*patch_size)
    patches = patches.permute(0, 2, 1, 3)  # (n, num_patches, c, patch_size*patch_size)
    patches = patches.reshape(
        batch_size, num_patches, channels * patch_size * patch_size
    )  # (n, num_patches, c*patch_size*patch_size)

    return patches


def patchify_stacked(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Converts a batch of stacked input images into patches.

    Parameters
    ----------
    images : torch.Tensor. Images tensor with shape (batch_size, num_imgs, channels, height, width).
    patch_size : int. Patch size in pixels; height and width must be multiples of patch_size.

    Returns
    -------
    torch.Tensor. Patches tensor with shape (batch_size, num_imgs, num_patches, channels * patch_size ** 2)
    where num_patches = (height / patch_size) * (width / patch_size).
    """
    # Extract shape, now including num_imgs
    batch_size, num_imgs, channels, height, width = images.shape

    # Assert conditions for height, width, and patch_size
    assert height % patch_size == 0, "Height is not a multiple of patch_size"
    assert width % patch_size == 0, "Width is not a multiple of patch_size"
    assert height == width, "Height and width are not equal"

    # Adjust unfold to handle the extra dimension by treating num_imgs as part of the batch dimension
    # This requires reshaping images to combine batch_size and num_imgs into a single dimension
    images_reshaped = images.reshape(batch_size * num_imgs, channels, height, width)
    patches = images_reshaped.unfold(2, patch_size, patch_size)
    patches = patches.unfold(3, patch_size, patch_size)

    # Calculate number of patches
    num_patches = (height // patch_size) * (width // patch_size)

    # Reshape to separate num_imgs from batch_size again and arrange dimensions to match output specification
    patches = patches.reshape(
        batch_size, num_imgs, channels, num_patches, patch_size * patch_size
    )
    patches = patches.permute(
        0, 1, 3, 2, 4
    )  # (batch_size, num_imgs, num_patches, channels, patch_size*patch_size)
    patches = patches.reshape(
        batch_size, num_imgs, num_patches, channels * patch_size * patch_size
    )

    return patches


def unpatchify(patches: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reconstructs images from a batch of patches.

    Parameters
    ----------
    patches : torch.Tensor. Patches tensor with shape (batch_size, num_patches, channels * patch_size ** 2).
    patch_size : int. Size of each patch in pixels.

    Returns
    -------
    torch.Tensor. Reconstructed images with shape (batch_size, channels, height, width).
    """
    n = patches.shape[0]
    c = patches.shape[2] / (patch_size * patch_size)
    assert c.is_integer()
    c = int(c)
    H = W = patch_size * int(patches.shape[1] ** 0.5)
    h = H // patch_size
    w = W // patch_size
    num_patches = patches.shape[1]

    patches = patches.reshape((n, num_patches, c, patch_size * patch_size))
    patches = patches.permute(0, 2, 1, 3)
    patches = patches.reshape((n, c, h, w, patch_size, patch_size))

    # Reordering the patches dimension back to the original form
    patches = patches.permute(
        0, 1, 2, 4, 3, 5
    ).contiguous()  # (n, c, patch_size, h, patch_size, w)

    # merging the patch dimensions back into the image dimensions
    images = patches.view(n, c, H, W)  # (n, c, H, W)

    return images


def unpatchify_stacked(
    patches: torch.Tensor, patch_size: int, num_imgs: int
) -> torch.Tensor:
    """Reconstructs images from their patches with an additional dimension for multiple images per batch.

    Parameters
    ----------
    patches : torch.Tensor. Patches tensor with shape (batch_size, num_imgs * num_patches, channels * patch_size ** 2).
    patch_size : int. Size of each patch in pixels.
    num_imgs : int. Number of images stacked in each batch.

    Returns
    -------
    torch.Tensor. Reconstructed images with shape (batch_size, num_imgs, channels, height, width).
    """
    n = patches.shape[0]
    # Calculate total number of patches per image
    total_patches = patches.shape[1] // num_imgs
    c = patches.shape[2] / (patch_size * patch_size)
    assert c.is_integer(), "Channels must be an integer value."
    c = int(c)
    H = W = patch_size * int(total_patches**0.5)
    h = H // patch_size
    w = W // patch_size

    # Reshape patches to reintroduce num_imgs dimension
    patches = patches.reshape((n, num_imgs, total_patches, c, patch_size * patch_size))
    patches = patches.permute(
        0, 1, 3, 2, 4
    )  # (n, num_imgs, c, total_patches, patch_size**2)
    patches = patches.reshape((n, num_imgs, c, h, w, patch_size, patch_size))

    # Reordering the patches dimension back to the original form
    patches = patches.permute(
        0, 1, 2, 3, 5, 4, 6
    ).contiguous()  # (n, num_imgs, c, patch_size, h, patch_size, w)

    # Merging the patch dimensions back into the image dimensions
    images = patches.view(n, num_imgs, c, H, W)  # (n, num_imgs, c, H, W)

    return images

def random_token_mask(
    size: tuple[int, int],
    mask_ratio: float = 0.6,
    mask_class_token: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Creates random token masks.

    Parameters
    ----------
    size : tuple[int, int]. (batch_size, sequence_length) of the token batch.
    mask_ratio : float. Fraction of tokens to mask (optional).
    mask_class_token : bool. If False the class token is never masked (optional).
    device : torch.device | str | None. Device on which to create the index masks (optional).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]. A (index_keep, index_mask) tuple where index_keep has
    shape (batch_size, num_keep) and index_mask has shape (batch_size, sequence_length - num_keep).
    """
    batch_size, sequence_length = size
    num_keep = int(sequence_length * (1 - mask_ratio))

    noise = torch.rand(batch_size, sequence_length, device=device)
    if not mask_class_token and sequence_length > 0:
        # make sure that class token is not masked
        noise[:, 0] = -1
        num_keep = max(1, num_keep)

    # get indices of tokens to keep
    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]

    return idx_keep, idx_mask

def image_token_mask(
    size: tuple[int, int],
    mask_images: int = 1,
    num_images: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Creates token mask to mask complete images.

    Parameters
    ----------
    size : tuple[int, int]. (batch_size, sequence_length) of the token batch.
    mask_images : int. Number of complete images to mask (optional).
    num_images : int. Total number of images in the sequence (optional).
    device : torch.device | str | None. Device on which to create the index masks (optional).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]. A (index_keep, index_mask) tuple where index_keep has
    shape (batch_size, num_keep) and index_mask has shape (batch_size, sequence_length - num_keep).
    """
    batch_size, sequence_length = size
    # Number of total image tokens
    num_img_tokens = sequence_length - 1
    # Number of tokens per image
    num_tokens_per_image = num_img_tokens // num_images

    assert mask_images < num_images, "Number of images to mask should be less than total number of images."

    # Create indeces of images to mask and keep
    img_idx = torch.arange(num_images, device=device).repeat(batch_size, 1)
    # Shuffle indeces of each row
    for i in range(batch_size):
        img_idx[i] = img_idx[i][torch.randperm(img_idx.size(1))]

    # Pick indeces of images to mask
    img_mask = img_idx[:, :mask_images]
    img_keep = img_idx[:, mask_images:]

    # Reshape indeces to pull out individual images
    token_idx = torch.arange(1, sequence_length, device=device).repeat(batch_size, 1)
    token_idx = token_idx.reshape(batch_size, num_images, -1) # (batch_size, num_images, num_tokens_per_image)

    # Get indeces of tokens to mask and keep
    idx_mask = token_idx[torch.arange(batch_size, device=device).unsqueeze(1), img_mask]
    idx_mask = idx_mask.reshape(batch_size, -1)
    idx_keep = token_idx[torch.arange(batch_size, device=device).unsqueeze(1), img_keep]
    idx_keep = idx_keep.reshape(batch_size, -1)

    # Class token will not be masked
    idx_keep = torch.cat([torch.zeros(batch_size, 1, dtype=torch.int, device=device), idx_keep], dim=1)

    return idx_keep, idx_mask

def nearest_neighbors(
    input_maps: torch.Tensor,
    candidate_maps: torch.Tensor,
    distances: torch.Tensor,
    num_matches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finds the nearest neighbors of the maps in input_maps in candidate_maps.

    Parameters
    ----------
    input_maps : torch.Tensor. Maps for which to find nearest neighbors,
        with shape (batch_size, input_map_size, feature_dimension).
    candidate_maps : torch.Tensor. Maps to search for nearest neighbors,
        with shape (batch_size, candidate_map_size, feature_dimension).
    distances : torch.Tensor. Pairwise distances between input and candidate maps,
        with shape (batch_size, input_map_size, candidate_map_size).
    num_matches : int. Number of nearest neighbors to return; if None or -1,
        all candidate maps are used.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]. A pair (filtered_input_maps, filtered_candidate_maps),
    each with shape (batch_size, num_matches, feature_dimension).
    """

    if num_matches is None or num_matches == -1 or num_matches > input_maps.size(1):
        num_matches = input_maps.size(1)

    # Find nearest neighbour of each input element in the candidate map
    topk_values, topk_indices = distances.topk(
        k=1, dim=2, largest=False
    )  # [bsz, input_map_size, 1]
    topk_values = topk_values.squeeze(-1)  # [bsz, input_map_size]

    # Select num_matches neighbors pairs having the lowest distance value.
    _, min_indices = topk_values.topk(
        k=num_matches, dim=1, largest=False
    )  # [bsz, num_matches]

    # Create the filtered input map with num_matches lowest distance values.
    feature_dimension = input_maps.shape[2]
    filtered_input_maps = torch.gather(
        input_maps, 1, min_indices.unsqueeze(-1).expand(-1, -1, feature_dimension)
    )  # [bsz, num_matches, feature_dimension]

    # Create candidate maps in the same way as input maps, but using corrispondent candidate values
    selected_candidate_maps = torch.gather(
        candidate_maps, 1, topk_indices.expand(-1, -1, feature_dimension)
    )  # [bsz, input_map_size, feature_dimension]
    filtered_candidate_maps = torch.gather(
        selected_candidate_maps,
        1,
        min_indices.unsqueeze(-1).expand(-1, -1, feature_dimension),
    )  # [bsz, num_matches, feature_dimension]

    return filtered_input_maps, filtered_candidate_maps


def get_weight_decay_parameters(
    modules: Iterable[Module],
    decay_batch_norm: bool = False,
    decay_bias: bool = False,
) -> tuple[list[Parameter], list[Parameter]]:
    """Returns all parameters of the modules that should be decayed and not decayed.

    Parameters
    ----------
    modules : Iterable[Module]. Modules from which to collect parameters.
    decay_batch_norm : bool. If True, batch norm parameters are weight-decayed (optional).
    decay_bias : bool. If True, bias parameters are weight-decayed (optional).

    Returns
    -------
    tuple[list[Parameter], list[Parameter]]. A (params, params_no_weight_decay) tuple.
    """
    params = []
    params_no_weight_decay = []
    for module in modules:
        for mod in module.modules():
            if isinstance(mod, _BatchNorm):
                if not decay_batch_norm:
                    params_no_weight_decay.extend(mod.parameters(recurse=False))
                else:
                    params.extend(mod.parameters(recurse=False))
            else:
                for name, param in mod.named_parameters(recurse=False):
                    if not decay_bias and name.endswith("bias"):
                        params_no_weight_decay.append(param)
                    else:
                        params.append(param)
    return params, params_no_weight_decay

