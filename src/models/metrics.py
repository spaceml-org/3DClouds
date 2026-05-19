import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.models.utils import get_profiles
from src.models.losses import PowerSpectrumLoss, MaskedSSIMLoss, DiceBCEFromContinuous, MaskedMSELoss


class PowerSpectrumMetric(nn.Module):
    """
    Computes the Power Spectrum loss as a metric (lower is better).
    """
    def __init__(self, mask=False):
        super().__init__()
        self.ps = PowerSpectrumLoss(coeff=1.0, mask=mask)  # coeff=1.0 for metric

    def forward(self, cs, cs_p, overpass_mask):
        return self.ps(cs, cs_p, overpass_mask)


class MaskedSSIMMetric(nn.Module):
    """
    Computes the Masked SSIM as a metric (higher is better).
    """
    def __init__(self, window_size=11, sigma=1.5, eps=1e-6):
        super().__init__()
        self.masked_ssim = MaskedSSIMLoss(window_size=window_size, sigma=sigma, mask=True, eps=eps)

    def forward(self, cs, cs_p, overpass_mask):
        # MaskedSSIMLoss returns a loss (1-SSIM), so invert it to get SSIM
        loss = self.masked_ssim(cs, cs_p, overpass_mask)
        return 1.0 - loss


class SSIMMetric(nn.Module):
    """
    SSIMMetric: Computes SSIMLoss to be provided to wandb as a metric.
    """

    def __init__(self):
        super().__init__()
        self.ssi = StructuralSimilarityIndexMeasure()

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, padded_length) (B, 90, 512) tensor
            cs_p: (batch_size, height, width, length) (B, 90, 256, 256) tensor
            overpass_mask: (batch_size, width, length) (B, 256, 256) tensor
        """
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)
        ssim_vals = []
        for i in range(len(cs)):
            cs_i = cs[i].unsqueeze(0).unsqueeze(0)
            cs_p_i = cs_p[i].unsqueeze(0).unsqueeze(0)
            ssim_vals.append(self.ssi(cs_p_i, cs_i))

        return torch.mean(torch.stack(ssim_vals))


class PSNRMetric(nn.Module):
    """
    PSNRMetric: Computes PSNR to be provided to wandb as a metric.
    """

    def __init__(self):
        super().__init__()
        self.pixel_max = 2  # Assuming the pixel values are normalized between -1 and 1

    def psnr(self, x_pred, targets):
        mse = torch.mean((targets - x_pred) ** 2)
        if mse == 0:
            return float("inf")
        PIXEL_MAX = self.pixel_max
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, padded_length) (B, 90, 512) tensor
            cs_p: (batch_size, height, width, length) (B, 90, 256, 256) tensor
            overpass_mask: (batch_size, width, length) (B, 256, 256) tensor
        """
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)
        psnr_vals = []
        for i in range(len(cs)):
            cs_i = cs[i].unsqueeze(0).unsqueeze(0)
            cs_p_i = cs_p[i].unsqueeze(0).unsqueeze(0)
            psnr_vals.append(self.psnr(cs_p_i, cs_i))

        return torch.mean(torch.stack(psnr_vals))


class DiceBCEFromContinuousMetric(nn.Module):
    """
    DiceBCEFromContinuousMetric: Computes the Dice and BCE loss as a metric.
    """

    def __init__(self, threshold=None, lambda_dice=1.0, lambda_bce=1.0):
        super().__init__()
        self.dice_bce = DiceBCEFromContinuous(
            threshold=threshold,
            lambda_dice=lambda_dice,
            lambda_bce=lambda_bce
        )

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) (4, 90, 128, 128) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """
        return self.dice_bce(cs, cs_p, overpass_mask)


class MaskedMSEMetric(nn.Module):
    """
    MaskedMSEMetric: Computes the masked MSE loss as a metric.
    """

    def __init__(self, clouds=True):
        super().__init__()
        self.masked_mse = MaskedMSELoss(clouds=clouds)

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) (4, 90, 128, 128) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """
        return self.masked_mse(cs, cs_p, overpass_mask)

# TODO: test if it works
# https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html
# LPIPS needs the images to be in the [-1, 1] range.
# Channels have to be 3, we have 1 (so we can use it 3 times)


# TODO: Check and update this
class PerceptualLossMetric(nn.Module):
    """
    Perceptual Loss: The Learned Perceptual Image Patch Similarity (LPIPS_)
    calculates perceptual similarity between two images.
    """

    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity()  # choose net_type

    def forward(self, cs, cs_p, overpass_mask):
        """
        Input:
            cs: (batch_size, height, width, length) (4, 90, 128, 128) tensor
            cs_p: (batch_size, n_channels, height, width) tensor
            overpass_mask: (batch_size, height, width) tensor
        """

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

            lpips_vals.append(self.lpips(rgb_image1, rgb_image2))

        mean_lpips = torch.mean(torch.stack(lpips_vals))
        return mean_lpips

