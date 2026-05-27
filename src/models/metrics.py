import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.models.utils import get_profiles
from src.models.losses import PowerSpectrumLoss, MaskedSSIMLoss, DiceBCEFromContinuous, MaskedMSELoss


class PowerSpectrumMetric(nn.Module):
    """ Power Spectrum metric (lower is better). """

    def __init__(self, mask=False):
        """ Initialize PowerSpectrumMetric.

            Parameters
            ----------
            mask : bool. Whether to apply masking in the underlying PowerSpectrumLoss (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.ps = PowerSpectrumLoss(coeff=1.0, mask=mask)  # coeff=1.0 for metric

    def forward(self, cs, cs_p, overpass_mask):
        """ Compute the power spectrum metric.

            Parameters
            ----------
            cs : torch.Tensor. Ground truth tensor.
            cs_p : torch.Tensor. Predicted tensor.
            overpass_mask : torch.Tensor. Overpass mask for profile extraction.

            Returns
            -------
            torch.Tensor. Power spectrum loss value (lower is better).
        """
        return self.ps(cs, cs_p, overpass_mask)


class MaskedSSIMMetric(nn.Module):
    """ Masked SSIM metric (higher is better). """

    def __init__(self, window_size=11, sigma=1.5, eps=1e-6):
        """ Initialize MaskedSSIMMetric.

            Parameters
            ----------
            window_size : int. Size of the Gaussian kernel (optional).
            sigma : float. Standard deviation of the Gaussian kernel (optional).
            eps : float. Small value to avoid division by zero (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.masked_ssim = MaskedSSIMLoss(window_size=window_size, sigma=sigma, mask=True, eps=eps)

    def forward(self, cs, cs_p, overpass_mask):
        """ Compute the masked SSIM metric.

            Parameters
            ----------
            cs : torch.Tensor. Ground truth tensor.
            cs_p : torch.Tensor. Predicted tensor.
            overpass_mask : torch.Tensor. Overpass mask for profile extraction.

            Returns
            -------
            torch.Tensor. Mean SSIM over valid pixels (higher is better).
        """
        # MaskedSSIMLoss returns a loss (1-SSIM), so invert it to get SSIM
        loss = self.masked_ssim(cs, cs_p, overpass_mask)
        return 1.0 - loss


class SSIMMetric(nn.Module):
    """ SSIM metric for comparing cloud profiles (higher is better). """

    def __init__(self):
        """ Initialize SSIMMetric. """
        super().__init__()
        self.ssi = StructuralSimilarityIndexMeasure()

    def forward(self, cs, cs_p, overpass_mask):
        """ Compute mean SSIM over extracted profiles.

            Parameters
            ----------
            cs : torch.Tensor. Ground truth of shape (B, H, padded_length).
            cs_p : torch.Tensor. Prediction of shape (B, H, W, L).
            overpass_mask : torch.Tensor. Overpass mask of shape (B, W, L).

            Returns
            -------
            torch.Tensor. Mean SSIM value across the batch.
        """
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)
        ssim_vals = []
        for i in range(len(cs)):
            cs_i = cs[i].unsqueeze(0).unsqueeze(0)
            cs_p_i = cs_p[i].unsqueeze(0).unsqueeze(0)
            ssim_vals.append(self.ssi(cs_p_i, cs_i))

        return torch.mean(torch.stack(ssim_vals))


class PSNRMetric(nn.Module):
    """ PSNR metric for evaluating reconstruction quality (higher is better). """

    def __init__(self):
        """ Initialize PSNRMetric. """
        super().__init__()
        self.pixel_max = 2  # Assuming the pixel values are normalized between -1 and 1

    def psnr(self, x_pred, targets):
        """ Compute Peak Signal-to-Noise Ratio between prediction and target.

            Parameters
            ----------
            x_pred : torch.Tensor. Predicted values.
            targets : torch.Tensor. Ground truth values.

            Returns
            -------
            float or torch.Tensor. PSNR value in dB; returns inf if MSE is zero.
        """
        mse = torch.mean((targets - x_pred) ** 2)
        if mse == 0:
            return float("inf")
        PIXEL_MAX = self.pixel_max
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

    def forward(self, cs, cs_p, overpass_mask):
        """ Compute mean PSNR over extracted profiles.

            Parameters
            ----------
            cs : torch.Tensor. Ground truth of shape (B, H, padded_length).
            cs_p : torch.Tensor. Prediction of shape (B, H, W, L).
            overpass_mask : torch.Tensor. Overpass mask of shape (B, W, L).

            Returns
            -------
            torch.Tensor. Mean PSNR value across the batch.
        """
        cs, cs_p = get_profiles(cs, cs_p, overpass_mask)
        psnr_vals = []
        for i in range(len(cs)):
            cs_i = cs[i].unsqueeze(0).unsqueeze(0)
            cs_p_i = cs_p[i].unsqueeze(0).unsqueeze(0)
            psnr_vals.append(self.psnr(cs_p_i, cs_i))

        return torch.mean(torch.stack(psnr_vals))


class DiceBCEFromContinuousMetric(nn.Module):
    """ Dice and BCE metric computed from continuous-valued predictions. """

    def __init__(self, threshold=None, lambda_dice=1.0, lambda_bce=1.0):
        """ Initialize DiceBCEFromContinuousMetric.

            Parameters
            ----------
            threshold : float or None. Binarisation threshold; uses per-patch minimum if None (optional).
            lambda_dice : float. Scaling coefficient for the Dice term (optional).
            lambda_bce : float. Scaling coefficient for the BCE term (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.dice_bce = DiceBCEFromContinuous(
            threshold=threshold,
            lambda_dice=lambda_dice,
            lambda_bce=lambda_bce
        )

    def forward(self, cs, cs_p, overpass_mask):
        """ Compute combined Dice and BCE metric.

            Parameters
            ----------
            cs : torch.Tensor. Ground truth of shape (B, H, W, L).
            cs_p : torch.Tensor. Prediction of shape (B, C, H, W).
            overpass_mask : torch.Tensor. Overpass mask of shape (B, H, W).

            Returns
            -------
            torch.Tensor. Combined Dice and BCE loss value.
        """
        return self.dice_bce(cs, cs_p, overpass_mask)


class MaskedMSEMetric(nn.Module):
    """ Masked MSE metric computed only over valid (cloud or clear-sky) pixels. """

    def __init__(self, clouds=True):
        """ Initialize MaskedMSEMetric.

            Parameters
            ----------
            clouds : bool. If True, evaluates on cloud pixels; if False, on clear-sky pixels (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.masked_mse = MaskedMSELoss(clouds=clouds)

    def forward(self, cs, cs_p, overpass_mask):
        """ Compute masked MSE metric.

            Parameters
            ----------
            cs : torch.Tensor. Ground truth of shape (B, H, W, L).
            cs_p : torch.Tensor. Prediction of shape (B, C, H, W).
            overpass_mask : torch.Tensor. Overpass mask of shape (B, H, W).

            Returns
            -------
            torch.Tensor. Mean MSE over valid pixels.
        """
        return self.masked_mse(cs, cs_p, overpass_mask)

# TODO: test if it works
# https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html
# LPIPS needs the images to be in the [-1, 1] range.
# Channels have to be 3, we have 1 (so we can use it 3 times)


# TODO: Check and update this
class PerceptualLossMetric(nn.Module):
    """ LPIPS perceptual similarity metric between cloud profile images. """

    def __init__(self):
        """ Initialize PerceptualLossMetric. """
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity()  # choose net_type

    def forward(self, cs, cs_p, overpass_mask):
        """ Compute mean LPIPS perceptual similarity over the batch.

            Parameters
            ----------
            cs : torch.Tensor. Ground truth of shape (B, H, W, L).
            cs_p : torch.Tensor. Prediction of shape (B, C, H, W).
            overpass_mask : torch.Tensor. Overpass mask of shape (B, H, W).

            Returns
            -------
            torch.Tensor. Mean LPIPS score across the batch.
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

