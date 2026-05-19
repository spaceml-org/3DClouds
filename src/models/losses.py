import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, MaskedDiceLoss  # , DiceCELoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn import HuberLoss
from src.models.utils import get_profiles

# Other ideas: histogram loss, gradient-based loss, ...


class MSELoss3D(nn.Module):
    """
    MSE Loss for 3D volumes

    Args:
        coeff (float): Scaling coefficient for SSIM loss.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, cs, cs_p, overpass_mask):
        masked = torch.stack(
            [torch.isfinite(cs[j, :]) for j in range(0, cs.shape[0])], axis=0
        )
        cs_masked = [
            torch.masked_select(cs[j, :], masked[j, :]).flatten()
            for j in range(0, cs.shape[0])
        ]
        cs_p_masked = [
            torch.masked_select(cs_p[j, :], masked[j, :]).flatten()
            for j in range(0, cs_p.shape[0])
        ]

        mse_loss = self.mse(torch.cat(cs_p_masked), torch.cat(cs_masked))
        return mse_loss


class MSELoss(nn.Module):
    """
    MSE loss for 2D profiles
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, padded_length) (B, 90, 512) tensor
            cs_p: (batch_size, height, width, length) (B, 90, 256, 256) tensor
            overpass_mask: (batch_size, width, length) (B, 256, 256) tensor
        """
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)
        mse_vals = []
        for i in range(
            len(cs)
        ):  # need to loop through since patches are not the same size
            mse_vals.append(self.mse(cs[i], cs_p[i]))
        mse = torch.mean(torch.stack(mse_vals))
        return mse


class CrossEntropyLoss(nn.Module):
    """
    Args:
        coeff (float): Scaling coefficient for SSIM loss.
    """

    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        self.CE = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=int(-1), weight=self.class_weights
        )

    def forward(self, cs, cs_p, overpass_mask):
        # NOTE: this relies on encoding of missing data being encoded as -1
        # The corresponding cloudsat transforms and also input_cloudsat dataset
        # (after the 3d expansion) should add -1's to missing data
        ce_loss = self.CE(cs_p, cs)
        return ce_loss


class DiceCrossEntropy(nn.Module):

    """
    SSIMLoss: Combines MSE and SSIM lossM.

    Args:
        coeff (float): Scaling coefficient for SSIM loss.
    """

    def __init__(
        self,
        class_weights_path=None,
        clear_sky_prop=0.25,
        lambda_dice=1.0,
        lambda_ce=1.0,
    ):
        # super(DiceCrossEntropy, self).__init__()
        super().__init__()
        self.class_weights_path = class_weights_path
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.clear_sky_prop = clear_sky_prop
        if self.class_weights_path is None:
            self.class_weights = None
        else:
            # this is the final weight we would like the clearsky class to have
            with open(self.class_weights_path) as json_data:
                cloudsat_props = json.load(json_data)
                json_data.close()
            cloudsat_props["proportions"][
                np.where(np.array(cloudsat_props["type"]) == 0)[0][0]
            ] = self.clear_sky_prop
            cloudsat_props = np.array(
                [
                    1 / p
                    for v, p in zip(
                        cloudsat_props["type"], cloudsat_props["proportions"]
                    )
                    if v != -1
                ]
            )
            cloudsat_props = cloudsat_props / np.sum(cloudsat_props)
            cloudsat_props = torch.Tensor(cloudsat_props)
            self.class_weights = cloudsat_props

        wsCE = torch.Tensor([self.class_weights[0], torch.sum(self.class_weights[1:])])
        wsDice = self.class_weights  # [1:]
        self.diceCE = MaskedDiceLoss(
            include_background=False,
            to_onehot_y=True,
            reduction="mean",
            weight=wsDice,
            softmax=True,
        )  # , lambda_dice=self.lambda_dice, lambda_ce=self.lambda_ce
        self.CE = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=int(-1), weight=wsCE
        )

    def forward(self, cs, cs_p, overpass_mask):
        # NOTE: this relies on encoding on missing data being encoded as -1
        # The corresponding cloudsat transforms and also input_cloudsat dataset
        # (after the 3d expansion) should add -1's to missing data
        bs, nc, h, ps, ps = cs_p.shape
        cs_p2 = torch.cat([torch.zeros(bs, 1, h, ps, ps).to(cs_p.device), cs_p], dim=1)
        cs2 = cs.unsqueeze(1) + 1
        mask = (
            cs2 >= 1
        ) * 1  # mask should be True for values you want to be taken into account
        dice_loss = self.diceCE(cs_p2, cs2, mask)
        max_cld, _ = torch.max(cs_p[:, 1:, :, :, :], dim=1)
        cs_p3 = torch.stack([cs_p[:, 0, :, :, :], max_cld]).permute(1, 0, 2, 3, 4)
        cs3 = (cs > 0) * 1
        cs3[cs == -1] = -1
        ce_loss = self.CE(cs_p3, cs3)
        dice_ce_loss = self.lambda_ce * ce_loss + self.lambda_dice * (dice_loss)
        return dice_ce_loss


class DiceBCEFromContinuous(nn.Module):
    """
    Computes Dice and BCE loss from continuous-valued targets and predictions,
    using a threshold to define cloud vs. clear regions.
    """

    def __init__(self, threshold=None, lambda_dice=1.0, lambda_bce=1.0):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.threshold = threshold  # If None, will use min value in cs per patch

    def forward(self, cs, cs_p, overpass_mask):
        """
        cs: ground truth, continuous (B, ...)
        cs_p: prediction, continuous (B, ...)
        overpass_mask: mask tensor
        """
        cs_list, cs_p_list = get_profiles(cs, cs_p, overpass_mask)
        dice_losses = []
        bce_losses = []

        for cs_i, cs_p_i in zip(cs_list, cs_p_list):
            cs_flat = cs_i.view(-1)
            cs_p_flat = cs_p_i.view(-1)
            min_val = cs_flat.min() if self.threshold is None else self.threshold
            valid_mask = (cs_flat != min_val)
            if valid_mask.sum() == 0:
                continue  # Skip this patch

            cs_bin = (cs_flat > min_val).float()[valid_mask]
            cs_p_bin = cs_p_flat[valid_mask]

            smooth = 1e-6
            probs = torch.sigmoid(cs_p_bin)
            denom = probs.sum() + cs_bin.sum() + smooth
            if denom == 0:
                dice_loss = torch.tensor(0.0, device=cs.device)
            else:
                intersection = (probs * cs_bin).sum()
                dice_loss = 1 - (2. * intersection + smooth) / denom

            if cs_bin.numel() == 0:
                bce_loss = torch.tensor(0.0, device=cs.device)
            else:
                bce_loss = self.bce(cs_p_bin, cs_bin)

            dice_losses.append(dice_loss)
            bce_losses.append(bce_loss)

        if len(dice_losses) == 0 or len(bce_losses) == 0:
            return torch.tensor(0.0, device=cs.device)

        # Average over all patches
        mean_dice = torch.mean(torch.stack(dice_losses))
        mean_bce = torch.mean(torch.stack(bce_losses))

        return self.lambda_dice * mean_dice + self.lambda_bce * mean_bce

class SSIMLoss(nn.Module):
    """
    SSIMLoss: SSIM loss.

    Args:
        coeff (float): Scaling coefficient for SSIM loss.
    """

    def __init__(self):
        super().__init__()
        self.ssi = StructuralSimilarityIndexMeasure(data_range=(-1, 1))

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) (4, 90, 128, 128) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)
        ssim_vals = []
        for i in range(
            len(cs)
        ):  # need to loop through since patches are not the same size
            ssim_vals.append(self.ssi(
                cs_p[i].unsqueeze(0).unsqueeze(0), cs[i].unsqueeze(0).unsqueeze(0)
            ))
        ssim = torch.mean(torch.stack(ssim_vals))

        ssim_loss = 1 - torch.mean(torch.stack(ssim_vals))

        return ssim_loss


class mse_ssimLoss(nn.Module):
    """
        SSIMLoss: Combines MSE and SSIM loss.

    Args:
        coeff (float): Scaling coefficient for ssim loss.
    """

    def __init__(self, coeff=1):
        super().__init__()
        self.coeff = coeff
        self.mse = MSELoss()
        self.ssim = SSIMLoss()

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) (4, 90, 128, 128) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """

        mse_loss = self.mse(cs, cs_p, overpass_mask)
        mean_ssim = self.ssim(cs, cs_p, overpass_mask)
        return mse_loss + self.coeff * mean_ssim


class ssim_mseLoss(nn.Module):
    """
        SSIMLoss: Combines MSE and SSIM loss.

    Args:
        coeff (float): Scaling coefficient for ssim loss.
    """

    def __init__(self, coeff=1):
        super().__init__()
        self.coeff = coeff
        self.mse = MSELoss()
        self.ssim = SSIMLoss()

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) (4, 90, 128, 128) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """

        mse_loss = self.mse(cs, cs_p, overpass_mask)
        mean_ssim = self.ssim(cs, cs_p, overpass_mask)
        return self.coeff * mse_loss + mean_ssim


class PerceptualLoss(nn.Module):
    """
    SSIMLoss: Combines MSE and Perceptual loss.

    Args:
        coeff (float): Scaling coefficient for LPIPS loss.
    """

    def __init__(self, coeff=1, net_type="alex"):
        super().__init__()
        self.coeff = coeff
        self.mse = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)
        self.minSize = 17
        if net_type == "alex":
            self.minSize = 64
        elif net_type == "vgg":
            self.minSize = 32

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) (4, 90, 128, 128) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """

        mse_loss = self.mse(cs, cs_p, overpass_mask)

        lpips_vals = []
        # create images for ssim
        batch_size, height, patch_size, _ = cs.shape
        for item in range(batch_size):
            cs_i = cs[item, :, :, :]
            cs_p_i = cs_p[item, :, :, :]
            overpass_mask_i = overpass_mask[item, :, :]

            binary_overpass_mask_i = overpass_mask_i > 0
            binary_overpass_mask_i = binary_overpass_mask_i.expand(height, -1, -1)

            cs_profile_i = cs_i[binary_overpass_mask_i].reshape([height, -1])
            cs_p_profile_i = cs_p_i[binary_overpass_mask_i].reshape([height, -1])

            # Replace NaN values with -1
            cs_profile_i[torch.isnan(cs_profile_i)] = -1
            cs_p_profile_i[torch.isnan(cs_p_profile_i)] = -1

            cs_profile_i = cs_profile_i.unsqueeze(0).unsqueeze(0)
            cs_p_profile_i = cs_p_profile_i.unsqueeze(0).unsqueeze(0)

            # Convert greyscale images to RGB
            rgb_image1 = cs_profile_i.repeat(1, 3, 1, 1)
            rgb_image2 = cs_p_profile_i.repeat(1, 3, 1, 1)

            # lpips_vals.append(self.lpips(rgb_image1, rgb_image2))
            # clamp values to -1 and 1 before passing to llpips

            if cs_profile_i.shape[3] >= self.minSize:
                val = self.lpips(
                    torch.clamp(rgb_image1, -1, 1), torch.clamp(rgb_image2, -1, 1)
                )
                lpips_vals.append(val)

        # Lower values indicate better perceptual similarity
        mean_lpips = torch.mean(torch.stack(lpips_vals))

        return mse_loss - self.coeff * mean_lpips


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss: Encourages smoothness in the output.

    Args:
        coeff (float): Scaling coefficient for TV loss.
    """

    def __init__(self, coeff=1):
        super().__init__()
        self.coeff = coeff

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """
        # Extract profiles from the volume
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)

        # Loss values
        loss_vals = []
        for i in range(len(cs)):  # need to loop through since patches are not the same size
            loss_vals.append(torch.sum(torch.abs(cs_p[i][:, :-1] - cs_p[i][:, 1:])) + \
                          torch.sum(torch.abs(cs_p[i][:-1, :] - cs_p[i][1:, :])))

        # Average TV loss over all patches
        tv_loss = torch.mean(torch.stack(loss_vals))

        return self.coeff * tv_loss


class HuberL1Loss(nn.Module):
    """
    Huber Loss: Combines MSE and MAE for robust regression.

    Args:
        delta (float): Threshold for switching between MSE and MAE.
        coeff (float): Scaling coefficient for Huber loss.
    """

    def __init__(self, delta=1.0, coeff=1.0):
        super().__init__()
        self.huber = HuberLoss(delta=delta)
        self.coeff = coeff

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """
        # Extract profiles from the volume
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)

        # Huber loss values
        loss_vals = []
        for i in range(len(cs)):  # need to loop through since patches are not the same size
            loss_vals.append(self.huber(cs_p[i], cs[i]))
        # Average Huber loss over all patches
        huber_loss = torch.mean(torch.stack(loss_vals))
        return self.coeff * huber_loss


class MaskedSSIMLoss(nn.Module):
    """
    Masked SSIM Loss: Computes SSIM over valid pixels defined by a binary mask.

    Args:
        window_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation for the Gaussian kernel.
        eps (float): Small value to avoid division by zero.
    """

    def __init__(self, window_size=11, sigma=1.5, eps=1e-6, coeff=1.0, mask=False):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.eps = eps
        self.coeff = coeff
        self.mask = mask

    def forward(self, cs, cs_p, overpass_mask):
        # Extract profiles from the volume
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)

        # Loss values
        ssim_vals = []
        for i in range(len(cs)):  # need to loop through since patches are not the same size
            # Compute mask from cs, with the mask being 0 where cs is the fill value (min)
            mask = (cs[i] != cs[i].min()).float().unsqueeze(0).unsqueeze(0) if self.mask else None  # Binary mask where valid pixels are 1
            ssim_vals.append(masked_ssim(
                cs_p[i].unsqueeze(0).unsqueeze(0), cs[i].unsqueeze(0).unsqueeze(0),
                mask=mask, window_size=self.window_size, sigma=self.sigma, eps=self.eps
            ))

        # Average SSIM over all patches
        mean_ssim = torch.mean(torch.stack(ssim_vals))
        # SSIM loss is 1 - SSIM
        ssim_loss = 1 - mean_ssim
        return self.coeff * ssim_loss



def gaussian_kernel(window_size: int, sigma: float, channels: int):
    """ Create a 2D Gaussian kernel.

    Args:
        window_size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.
        channels (int): Number of channels in the input tensor.
    Returns:
        kernel_2d (torch.Tensor): 2D Gaussian kernel of shape (channels, 1, window_size, window_size).
    """

    # Ensure window_size is odd
    coords = torch.arange(window_size).float() - window_size // 2
    # Create a 1D Gaussian kernel
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # Create a 2D Gaussian kernel by outer product
    kernel_2d = g[:, None] @ g[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size)
    return kernel_2d

def masked_ssim(pred, target, mask=None, window_size=11, sigma=1.5, eps=1e-6):
    """
    Compute SSIM over valid pixels defined by a binary mask.
    pred, target: (B, C, H, W)
    mask: (B, 1, H, W) binary mask (1 = valid, 0 = invalid)
    """
    B, C, H, W = pred.shape
    device = pred.device

    # Gaussian kernel
    kernel = gaussian_kernel(window_size, sigma, C).to(device)

    # If mask is None, create a mask of ones (valid everywhere)
    if mask is None:
        mask = torch.ones((B, 1, H, W), device=device)

    # Ensure mask is float
    mask = mask.float()

    # Convolve mask to get normalization factor per patch
    mask_conv = F.conv2d(mask, kernel, padding=window_size // 2, groups=1)
    mask_conv = mask_conv.clamp(min=eps)

    def filtered_mean(x):
        return F.conv2d(x * mask, kernel, padding=window_size // 2, groups=C) / mask_conv

    mu_x = filtered_mean(pred)
    mu_y = filtered_mean(target)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    def filtered_var(x, mu):
        return F.conv2d((x ** 2) * mask, kernel, padding=window_size // 2, groups=C) / mask_conv - mu ** 2

    sigma_x_sq = filtered_var(pred, mu_x)
    sigma_y_sq = filtered_var(target, mu_y)
    sigma_xy = F.conv2d((pred * target) * mask, kernel, padding=window_size // 2, groups=C) / mask_conv - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    # Average SSIM over valid regions only
    valid_mean = (ssim_map * mask).sum(dim=[1, 2, 3]) / mask.sum(dim=[1, 2, 3]).clamp(min=eps)
    return valid_mean.mean()


class PowerSpectrumLoss(nn.Module):
    """
    Power Spectrum Loss: Computes the power spectrum loss between two 3D tensors.
    Optionally handles masking by filling masked regions with the mean of valid values.

    Args:
        coeff (float): Scaling coefficient for the power spectrum loss.
        mask (bool): Whether to apply masking logic.
    """

    def __init__(self, coeff=1.0, mask=False):
        super().__init__()
        self.coeff = coeff
        self.mask = mask

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        Output:
            Power spectrum loss value.
        """
        # Extract profiles from the volume
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)

        ps_loss = []

        for i in range(len(cs)):
            target = cs[i]
            pred = cs_p[i]

            if self.mask:
                mask = (target != target.min()).float()
                if mask.sum() == 0:
                    continue  # skip fully masked patch

                target = target * mask
                pred = pred * mask

            ps_target = torch.log1p(torch.abs(torch.fft.fftn(target)) ** 2)
            ps_pred = torch.log1p(torch.abs(torch.fft.fftn(pred)) ** 2)

            ps_loss.append(torch.mean((ps_target - ps_pred) ** 2))

        if len(ps_loss) == 0:
            return torch.tensor(0.0, device=cs[0].device if len(cs) > 0 else "cpu")
        return self.coeff * torch.mean(torch.stack(ps_loss))

class MaskedMSELoss(nn.Module):
    """
    Masked MSE loss for 2D profiles, only over valid (finite) values.
    """

    def __init__(self, clouds=True):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.clouds = clouds  # If True, assumes cs is cloudsat profile

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, padded_length) (B, 90, 512) tensor
            cs_p: (batch_size, height, width, length) (B, 90, 256, 256) tensor
            overpass_mask: (batch_size, width, length) (B, 256, 256) tensor
        """
        cs_list, cs_p_list = get_profiles(cs, cs_p, overpass_mask)
        mse_vals = []
        for cs_i, cs_p_i in zip(cs_list, cs_p_list):
            min_val = cs_i.min()
            if self.clouds:
                mask = torch.isfinite(cs_i) & (cs_i != min_val)
            else:
                mask = torch.isfinite(cs_i) & (cs_i <= min_val)
            if mask.sum() == 0:
                continue  # skip if no valid values
            cs_valid = cs_i[mask]
            cs_p_valid = cs_p_i[mask]
            mse_vals.append(self.mse(cs_p_valid, cs_valid))
        if len(mse_vals) == 0:
            return torch.tensor(0.0, device=cs.device)
        mse = torch.mean(torch.stack(mse_vals))
        return mse


class HybridLoss(nn.Module):
    """
    Hybrid Loss: Combines Huber loss, SSIM loss, Masked SSIM, and TV loss.
    Only computes each loss if its coefficient is > 0.
    """

    def __init__(self, mse_coeff=1.0,
                 huber_delta=1.0, huber_coeff=0.,
                 ssim_coeff=0., masked_ssim_coeff=0.,
                 ssim_window_size=11, ssim_sigma=1.5, ssim_eps=1e-6,
                 masked_ssim_window_size=11, masked_ssim_sigma=1.5, masked_ssim_eps=1e-6,
                 tv_coeff=0., ps_coeff=0.0, ps_mask=False, dice_coeff=0.0,
                 bce_coeff=0.0, masked_mse_clouds_coeff=0.0, masked_mse_clearsky_coeff=0.0):
        super().__init__()

        # Weights
        self.mse_coeff = mse_coeff
        self.huber_coeff = huber_coeff
        self.ssim_coeff = ssim_coeff
        self.masked_ssim_coeff = masked_ssim_coeff
        self.tv_coeff = tv_coeff
        self.ps_coeff = ps_coeff
        self.ps_mask = ps_mask
        self.dice_coeff = dice_coeff
        self.bce_coeff = bce_coeff
        self.masked_mse_clouds_coeff = masked_mse_clouds_coeff
        self.masked_mse_clearsky_coeff = masked_mse_clearsky_coeff

        # Loss functions
        if mse_coeff > 0:
            self.mse = MSELoss()
        if huber_coeff > 0:
            self.huber = HuberL1Loss(delta=huber_delta)
        if ssim_coeff > 0:
            self.ssim = MaskedSSIMLoss(window_size=ssim_window_size,
                                       sigma=ssim_sigma, mask=False,
                                       eps=ssim_eps)
        if masked_ssim_coeff > 0:
            self.masked_ssim = MaskedSSIMLoss(window_size=masked_ssim_window_size,
                                              sigma=masked_ssim_sigma, mask=True,
                                              eps=masked_ssim_eps)
        if tv_coeff > 0:
            self.tv = TotalVariationLoss()
        if ps_coeff > 0:
            self.ps = PowerSpectrumLoss(coeff=ps_coeff, mask=ps_mask)
        if dice_coeff > 0:
            self.dice = DiceBCEFromContinuous(lambda_dice=1, lambda_bce=0)
        if bce_coeff > 0:
            self.bce = DiceBCEFromContinuous(lambda_dice=0, lambda_bce=1)
        if masked_mse_clouds_coeff > 0:
            self.masked_mse_clouds = MaskedMSELoss(clouds=True)
        if masked_mse_clearsky_coeff > 0:
            self.masked_mse_clearsky = MaskedMSELoss(clouds=False)

    def forward(self, cs, cs_p, overpass_mask):

        # Hybrid loss term
        hybrid_loss = 0.0

        # Compute each loss term if its coefficient is > 0
        if self.mse_coeff > 0:
            hybrid_loss += self.mse(cs, cs_p, overpass_mask) * self.mse_coeff
        if self.huber_coeff > 0:
            hybrid_loss += self.huber(cs, cs_p, overpass_mask) * self.huber_coeff
        if self.ssim_coeff > 0:
            hybrid_loss += self.ssim(cs, cs_p, overpass_mask) * self.ssim_coeff
        if self.masked_ssim_coeff > 0:
            hybrid_loss += self.masked_ssim(cs, cs_p, overpass_mask) * self.masked_ssim_coeff
        if self.tv_coeff > 0:
            hybrid_loss += self.tv(cs, cs_p, overpass_mask) * self.tv_coeff
        if self.ps_coeff > 0:
            hybrid_loss += self.ps(cs, cs_p, overpass_mask) * self.ps_coeff
        if self.dice_coeff > 0:
            hybrid_loss += self.dice(cs, cs_p, overpass_mask) * self.dice_coeff
        if self.bce_coeff > 0:
            hybrid_loss += self.bce(cs, cs_p, overpass_mask) * self.bce_coeff
        if self.masked_mse_clouds_coeff > 0:
            hybrid_loss += self.masked_mse_clouds(cs, cs_p, overpass_mask) * self.masked_mse_clouds_coeff
        if self.masked_mse_clearsky_coeff > 0:
            hybrid_loss += self.masked_mse_clearsky(cs, cs_p, overpass_mask) * self.masked_mse_clearsky_coeff

        return hybrid_loss


