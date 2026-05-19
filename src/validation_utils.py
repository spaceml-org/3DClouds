"""
This file contains functionality for creating validation plots which can be shared
across experiments and models.
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score

import wandb
from src.models.losses import MSELoss
from src.models.metrics import (
    DiceBCEFromContinuousMetric,
    MaskedMSEMetric,
    MaskedSSIMMetric,
    PerceptualLossMetric,
    PowerSpectrumMetric,
    PSNRMetric,
    SSIMMetric,
)
from src.models.utils import get_profiles


def plot_profiles(
    x: np.ndarray,
    cs: np.ndarray,
    cs_p: np.ndarray,
    overpass_mask: np.ndarray,
    current_epoch: int,
    log_image_samples,
    experiment,
    plot_channel,
    satellite=None,
    task="regression",
    batch_idx="",
):
    """
    Function to plot the true and predicted cloudsat profiles and upload them to wandb.

    Inputs:
        x: batch of input images (modis or msg) [batch_size x n_channels x patch_size_x x patch_size_y]
        cs: batch of true 3d cloudsat profiles [batch_size x height x patch_size_x x patch_size_y]
        cs_p: batch of predicted 3d cloudsat profiles [batch_size x height x patch_size_x x patch_size_y]
        overpass_mask: batch of overpass masks [batch_size x patch_size_x x patch_size_y] # TODO check size
        current_epoch: the current epoch, used for labelling the plot when uploaded to wandb
        log_image_samples: number of images to plot
        experiment: wandb experiment
        plot_channel: the input image channel that is supposed to be plotted
    """

    # check that we are not trying to plot more samples than we have in a batch
    batch_size = x.shape[0]
    if log_image_samples > batch_size:
        log_image_samples = batch_size

    # plot masked input, prediction, and original image
    fig, axes = plt.subplots(log_image_samples, 3, figsize=(10, log_image_samples * 3))

    # set colormaps
    overpass_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "", [(0, 0, 0, 0), (1, 0, 0, 1)]
    )  # transparent to red
    if task == "regression-radar_reflectivity":
        profile_cmap = "BuGn"  # white over green to blue
    elif task == "regression-iwc":
        profile_cmap = "coolwarm"
    elif task == "regression-re":
        profile_cmap = "BuPu"
    elif task == "regression-qr":
        profile_cmap = "RdBu_r"
    elif task == "segmentation":
        profile_cmap = "Dark2"
    else:
        profile_cmap = "viridis"

    image_cmap = "Blues"

    cs, cs_p = get_profiles(cs, cs_p, overpass_mask)

    for i in range(log_image_samples):
        x_i = x[i].cpu().numpy() if x[i].is_cuda else x[i]
        cs_i = cs[i].cpu().numpy() if cs[i].is_cuda else cs[i]
        cs_p_i = cs_p[i].cpu().numpy() if cs_p[i].is_cuda else cs_p[i]
        overpass_mask_i = (
            overpass_mask[i].cpu().numpy()
            if overpass_mask[i].is_cuda
            else overpass_mask[i]
        )

        vmin = np.nanmin(x_i[plot_channel])
        vmax = np.nanmax(x_i[plot_channel])

        # plot one of the modis input channels
        axes[i, 0].imshow(
            x_i[plot_channel],
            cmap=image_cmap,
            vmin=vmin,  # the input images are normalized to [-1, 1]
            vmax=vmax,  # lock the colorbar to the same range for all images
            interpolation=None,
        )

        # binarise overpass mask
        overpass_i = (overpass_mask_i > 0).squeeze()

        # plot as red overlay
        axes[i, 0].imshow(
            overpass_i, vmin=0, vmax=1, cmap=overpass_cmap, interpolation="nearest"
        )

        # lock the colorbar to the same range
        vmin = -1  # np.nanmin([min_cs_p_i, min_cs_i])
        vmax = 1  # np.nanmax([max_cs_p_i, max_cs_i])

        # plot profiles and add shared colorbar
        axes[i, 1].imshow(cs_p_i, vmin=vmin, vmax=vmax, cmap=profile_cmap)
        im_cs = axes[i, 2].imshow(cs_i, vmin=vmin, vmax=vmax, cmap=profile_cmap)
        fig.colorbar(im_cs, ax=axes[i, 2], fraction=0.046, pad=0.04)

    # # set titles for clarity
    if satellite is not None:
        axes[0, 0].set_title(f"{satellite[0].split('_')[0]} channel {plot_channel}")
    else:
        axes[0, 0].set_title(f"Input channel {plot_channel}")
    axes[0, 1].set_title("Predicted Cloudsat Profile")
    axes[0, 2].set_title("True Cloudsat Profile")

    fig.suptitle(batch_idx)
    # upload to wandb
    experiment.log({f"images/{current_epoch:02d}_og_pred_true": wandb.Image(fig)})
    plt.close(fig)


def plot_multi_profiles(
    x: np.ndarray,
    cs: dict[np.ndarray],
    cs_p: np.ndarray,
    overpass_mask: np.ndarray,
    current_epoch: int,
    log_image_samples,
    experiment,
    plot_channel,
    satellite=None,
    task="regression",
    batch_idx="",
):
    """
    Function to plot the true and predicted cloudsat profiles and upload them to wandb for multiple output variables

    Inputs:
        x: batch of input images (modis or msg) [batch_size x n_channels x patch_size_x x patch_size_y]
        cs: batch of true 3d cloudsat profiles [batch_size x height x patch_size_x x patch_size_y]
        cs_p: batch of predicted 3d cloudsat profiles [batch_size x vars x height x patch_size_x x patch_size_y]
        overpass_mask: batch of overpass masks [batch_size x patch_size_x x patch_size_y] # TODO check size
        current_epoch: the current epoch, used for labelling the plot when uploaded to wandb
        log_image_samples: number of images to plot
        experiment: wandb experiment
        plot_channel: the input image channel that is supposed to be plotted
    """

    # check that we are not trying to plot more samples than we have in a batch
    batch_size = x.shape[0]
    if log_image_samples > batch_size:
        log_image_samples = batch_size

    n_variables = len(cs)
    figs_wide = 2 * n_variables + 1

    # Check if output prediction shape is wrong for single variable prediction, unsqueeze for compatibility:
    if n_variables == 1 and len(cs_p.shape) < 5:
        cs_p = cs_p.unsqueeze(1)

    # plot masked input, prediction, and original image
    fig, axes = plt.subplots(
        log_image_samples, figs_wide, figsize=(3.2 * figs_wide, log_image_samples * 2.4)
    )

    # set colormaps
    overpass_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "", [(0, 0, 0, 0), (1, 0, 0, 1)]
    )  # transparent to red

    image_cmap = "Blues"
    cmap_dict = {
        "radar_reflectivity": "BuGn",
        "iwc": "PuBu",
        "re": "RdPu",
    }

    cmap = lambda var: cmap_dict.get(var.lower(), "viridis")

    for i in range(log_image_samples):
        x_i = x[i].cpu().numpy() if x[i].is_cuda else x[i]
        overpass_mask_i = (
            overpass_mask[i].cpu().numpy()
            if overpass_mask[i].is_cuda
            else overpass_mask[i]
        )

        vmin = np.nanmin(x_i[plot_channel])
        vmax = np.nanmax(x_i[plot_channel])

        # plot one of the modis input channels
        axes[i, 0].imshow(
            x_i[plot_channel],
            cmap=image_cmap,
            vmin=vmin,  # the input images are normalized to [-1, 1]
            vmax=vmax,  # lock the colorbar to the same range for all images
            interpolation=None,
        )

        # binarise overpass mask
        overpass_i = (overpass_mask_i > 0).squeeze()

        # plot as red overlay
        axes[i, 0].imshow(
            overpass_i, vmin=0, vmax=1, cmap=overpass_cmap, interpolation="nearest"
        )

    # lock the colorbar to the same range
    vmin = -1  # np.nanmin([min_cs_p_i, min_cs_i])
    vmax = 1  # np.nanmax([max_cs_p_i, max_cs_i])

    # Now loop over cloudsat vars
    for k, (j, var) in enumerate(zip(range(1, 2 * n_variables, 2), cs.keys())):
        cs_j, cs_pj = get_profiles(cs[var].transpose(-2, -1), cs_p[:, k], overpass_mask)
        # plot profiles and add shared colorbar
        for i in range(log_image_samples):
            axes[i, j].imshow(
                cs_pj[i].cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap(var)
            )
            im_cs = axes[i, j + 1].imshow(
                cs_j[i].cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap(var)
            )
            plt.colorbar(
                im_cs,
                ax=axes[i, j : j + 2],
                shrink=0.6,
                pad=0.2,
                orientation="horizontal",
                label=var,
            )
        axes[0, j].set_title(f"Predicted profile")
        axes[0, j + 1].set_title(f"Target profile")

    # # set titles for clarity
    if satellite is not None:
        axes[0, 0].set_title(f"{satellite[0].split('_')[0]} channel {plot_channel}")
    else:
        axes[0, 0].set_title(f"Input channel {plot_channel}")

    fig.suptitle(batch_idx)
    # upload to wandb
    experiment.log({f"images/{current_epoch:02d}_og_pred_true": wandb.Image(fig)})
    plt.close(fig)


def plot_3d_prediction(
    x_i: np.ndarray,
    cs_i: np.ndarray,
    cs_p_i: np.ndarray,
    current_epoch: int,
    experiment,
    plot_channel,
    remove_edge_rows=30,
):
    """
    Function to add a 3D plot of the true cloudsat profile and the predicted clouds to wandb.

    Inputs:
        x_i: image (modis or msg) input (n_channels x patch_size_x x patch_size_y)
        cs_i: true 3d cloudsat profile (height x patch_size_x x patch_size_y)
        cs_p_i: predicted 3d cloudsat profile (height x patch_size_x x patch_size_y)
        current_epoch: the current epoch, used for labelling the plot when uploaded to wandb
        experiment: wandb experiment
        plot_channel: the input image channel that is supposed to be plotted
        remove_edge_rows: number of rows to remove from the edge of the 3D plot

    """

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    axs = [ax1, ax2]

    # check that image channel is valid, if too large, set to last channel
    if plot_channel > x_i.shape[0]:
        plot_channel = x_i.shape[0]
    elif plot_channel < 0:
        plot_channel = 0

    # get dimensions of 3D cube
    xmax, ymax = x_i[plot_channel].shape
    zmax = cs_i.shape[0]
    xmax, ymax, zmax

    # create 2d and 3d meshgrids for plotting
    x_linspace = np.linspace(0, xmax, xmax)
    y_linspace = np.linspace(0, ymax, ymax)
    z_linspace = np.linspace(0, zmax, zmax)

    xmesh_3d, ymesh_3d, zmesh_3d = np.meshgrid(x_linspace, y_linspace, z_linspace)

    xmesh_2d, ymesh_2d = np.meshgrid(x_linspace, y_linspace)
    zmesh_2d = np.zeros_like(xmesh_2d)

    # the overpass and backwards (so we can see the prediction at the overpass
    # location and what's happening 'behind' it)
    for ax, cloudsat in zip(axs, [cs_i, cs_p_i]):
        # prepare cloudsat profile for 3D plot
        cloudsat = cloudsat.transpose(1, 2, 0)

        cloudsat[
            cloudsat < -0.95
        ] = (
            np.nan
        )  # set values below -0.95 to nan in cloudsat_tr so they are not shown in plot

        # get nan indices from cloudsat_tr
        valid_indices = np.where(
            ~np.isnan(cloudsat.squeeze()[:, remove_edge_rows : -1 * remove_edge_rows])
        )

        # get points that we want to plot from valid indices
        cloudsat_plot = cloudsat.squeeze()[valid_indices]
        XX_plot = xmesh_3d[:, remove_edge_rows : -1 * remove_edge_rows][valid_indices]
        YY_plot = ymesh_3d[:, remove_edge_rows : -1 * remove_edge_rows][valid_indices]
        ZZ_plot = zmesh_3d[:, remove_edge_rows : -1 * remove_edge_rows][valid_indices]

        # Creating plot
        ax.plot_surface(
            xmesh_2d,
            ymesh_2d,
            zmesh_2d,
            rstride=2,
            cstride=2,
            facecolors=plt.cm.viridis(x_i[plot_channel] + -1 * x_i[plot_channel].min()),
            alpha=1,
        )
        ax.scatter3D(
            XX_plot, YY_plot, ZZ_plot, c=cloudsat_plot, cmap="viridis", alpha=0.2
        )
        plt.title("Cloudsat 3D scatter plot")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_zlim(0, zmax)

        # rotate the plot
        ax.view_init(15, 160)

        # reverse the axis
        ax.invert_zaxis()

    # upload to wandb
    experiment.log({f"images/{current_epoch}_3d_plot": wandb.Image(fig)})


def plot_profiles_from_2d(
    cs: np.ndarray,
    cs_t: np.ndarray,
    cs_p: np.ndarray,
    current_epoch: int,
    log_image_samples,
    experiment,
    vmin=None,
):
    """
    Function to plot the true and predicted cloudsat profiles and upload them to wandb.

    Used in unconditioned diffusion experiments.

    Inputs:
        x: batch of input images (modis or msg) [batch_size x n_channels x patch_size_x x patch_size_y]
        cs: batch of true 3d cloudsat profiles [batch_size x height x patch_size_x x patch_size_y]
        cs_p: batch of predicted 3d cloudsat profiles [batch_size x height x patch_size_x x patch_size_y]
        overpass_mask: batch of overpass masks [batch_size x patch_size_x x patch_size_y]
        current_epoch: the current epoch, used for labelling the plot when uploaded to wandb
        log_image_samples: number of images to plot
        experiment: wandb experiment
        plot_channel: the input image channel that is supposed to be plotted
    """

    # check that we are not trying to plot more samples than we have in a batch
    batch_size = cs.shape[0]
    if log_image_samples > batch_size:
        log_image_samples = batch_size

    # plot masked input, prediction, and original image
    fig, axes = plt.subplots(log_image_samples, 3, figsize=(10, log_image_samples * 2))
    if log_image_samples == 1:
        axes = np.expand_dims(axes, axis=0)
    # set colormaps
    profile_cmap = "BuGn"  # white over green to blue

    for i in range(log_image_samples):
        # plot one of the modis input channels
        if not vmin:
            # TODO: Add clipping of outliers?
            vmin = np.nanmin([np.nanmin(cs_p[i]), np.nanmin(cs[i])])
        vmax = np.nanmax([np.nanmax(cs_p[i]), np.nanmax(cs[i])])

        axes[i, 0].imshow(cs[i].squeeze().T, cmap=profile_cmap, vmin=vmin, vmax=vmax)

        axes[i, 1].imshow(cs_t[i].squeeze().T, cmap=profile_cmap, vmin=vmin, vmax=vmax)

        im_cs = axes[i, 2].imshow(
            cs_p[i].squeeze().T, cmap=profile_cmap, vmin=vmin, vmax=vmax
        )

        fig.colorbar(im_cs, ax=axes[i, 2], fraction=0.046, pad=0.04)

    # set titles for clarity
    axes[0, 0].set_title(r"True Cloudsat Profile $cs$")
    axes[0, 1].set_title(r"Noisy Cloudsat Profile $cs_t$")
    axes[0, 2].set_title(r"Denoised Cloudsat Profile $cs_p$")

    # upload to wandb
    experiment.log({f"images/{current_epoch}_pred_true": wandb.Image(fig)})
    plt.close(fig)


def ssim_metric(x_pred, targets, overpass_mask):
    SSIM = SSIMMetric().to("cuda")
    ssi_cs = SSIM(cs=targets, cs_p=x_pred, overpass_mask=overpass_mask)
    return ssi_cs


def psnr_metric(x_pred, targets, overpass_mask):
    PSNR = PSNRMetric().to("cuda")
    psnr_cs = PSNR(cs=targets, cs_p=x_pred, overpass_mask=overpass_mask)
    return psnr_cs


def lpips_metric(x_pred, targets, overpass_mask):
    LPIPS = PerceptualLossMetric().to("cuda")
    lpips_cs = LPIPS(cs=targets, cs_p=x_pred, overpass_mask=overpass_mask)
    return lpips_cs


def power_spectrum_metric(x_pred, targets, overpass_mask):
    PowerSpectrum = PowerSpectrumMetric(mask=False).to("cuda")
    power_spectrum_cs = PowerSpectrum(
        cs=targets, cs_p=x_pred, overpass_mask=overpass_mask
    )
    return power_spectrum_cs


def masked_power_spectrum_metric(x_pred, targets, overpass_mask):
    PowerSpectrum = PowerSpectrumMetric(mask=True).to("cuda")
    power_spectrum_cs = PowerSpectrum(
        cs=targets, cs_p=x_pred, overpass_mask=overpass_mask
    )
    return power_spectrum_cs


def dice_metric(x_pred, targets, overpass_mask):
    DiceBCE = DiceBCEFromContinuousMetric(lambda_dice=1, lambda_bce=0).to("cuda")
    dice = DiceBCE(cs=targets, cs_p=x_pred, overpass_mask=overpass_mask)
    return dice


def bce_metric(x_pred, targets, overpass_mask):
    DiceBCE = DiceBCEFromContinuousMetric(lambda_dice=0, lambda_bce=1).to("cuda")
    bce = DiceBCE(cs=targets, cs_p=x_pred, overpass_mask=overpass_mask)
    return bce


def mssim_metric(x_pred, targets, overpass_mask):
    MSSIM = MaskedSSIMMetric().to("cuda")
    mssim_cs = MSSIM(cs=targets, cs_p=x_pred, overpass_mask=overpass_mask)
    return mssim_cs


def masked_mse_metric(x_pred, targets, overpass_mask, clouds=True):
    MaskedMSE = MaskedMSEMetric(clouds=clouds).to("cuda")
    mse_cs = MaskedMSE(cs=targets, cs_p=x_pred, overpass_mask=overpass_mask)
    return mse_cs


def log_metrics_wandb(
    cs,
    cs_p,
    overpass_mask,
    stage,
    experiment,
):
    """
    Function to log metrics to wandb.
    cs: true cloudsat profile
    cs_p: predicted cloudsat profile
    overpass_mask: overpass mask
    stage: train, val or test
    experiment: wandb experiment
    """
    mse = MSELoss().to("cuda")
    mse_cs = mse(cs, cs_p, overpass_mask)

    # Normalized RMSE
    rmse_cs = torch.sqrt(mse_cs)

    # Structural Similarity Index (SSIM)
    ssi_cs = ssim_metric(
        x_pred=cs_p,
        targets=cs,
        overpass_mask=overpass_mask,
    )

    # Log PSNR
    psnr_cs = psnr_metric(x_pred=cs_p, targets=cs, overpass_mask=overpass_mask)

    # MSSIM
    mssim_cs = mssim_metric(x_pred=cs_p, targets=cs, overpass_mask=overpass_mask)
    # PS
    ps = power_spectrum_metric(x_pred=cs_p, targets=cs, overpass_mask=overpass_mask)
    # Masked PS
    mps = masked_power_spectrum_metric(
        x_pred=cs_p, targets=cs, overpass_mask=overpass_mask
    )
    # Dice BCE from continuous
    dice = dice_metric(x_pred=cs_p, targets=cs, overpass_mask=overpass_mask)
    bce = bce_metric(x_pred=cs_p, targets=cs, overpass_mask=overpass_mask)
    # Masked MSE
    cloud_mse = masked_mse_metric(
        x_pred=cs_p, targets=cs, overpass_mask=overpass_mask, clouds=True
    )
    clear_mse = masked_mse_metric(
        x_pred=cs_p, targets=cs, overpass_mask=overpass_mask, clouds=False
    )

    # Log metrics to wandb
    experiment.log({f"{stage}/mse (global)": mse_cs.item()})
    experiment.log({f"{stage}/rmse (global)": rmse_cs.item()})
    experiment.log({f"{stage}/pnsr (global)": psnr_cs.item()})
    experiment.log({f"{stage}/ssim (global)": ssi_cs.item()})
    experiment.log({f"{stage}/masked_ssim (clouds only)": mssim_cs.item()})
    experiment.log({f"{stage}/power spectrum (clouds only)": mps.item()})
    experiment.log({f"{stage}/dice (cloud masks)": dice.item()})
    # experiment.log({f"{stage}/bce (cloud masks)": bce.item()})
    experiment.log({f"{stage}/masked_mse (clouds only)": cloud_mse.item()})
    experiment.log({f"{stage}/masked_mse (excluding clouds)": clear_mse.item()})

    # Return so that Pytorch model has access to these metrics for checkpointing
    return (
        mse_cs,
        rmse_cs,
        ssi_cs,
        psnr_cs,
        mssim_cs,
        mps,
        dice,
        bce,
        cloud_mse,
        clear_mse,
    )


def log_metrics_wandb_seg(
    cs, cs_p, overpass_mask, stage, experiment, num_classes=9, class_weights=None
):
    """
    Function to log segmentation metrics to wandb.
    cs: true cloudsat cloud types
    cs_p: estimated cloudsat probabilities per class
    overpass_mask: overpass mask
    stage: train, val or test
    experiment: wandb experiment
    """
    # Check for finite values in target
    mask = cs != -1
    mask_cs = cs[mask]

    cs_p_class = torch.argmax(cs_p, dim=1)
    # mask cs_p_class
    mask_cs_p_class = cs_p_class[mask]

    # Structural Similarity Index (SSIM)
    ssi_cs = ssim_metric(
        x_pred=cs_p_class.to(torch.float64),
        targets=cs.to(torch.float64),
        overpass_mask=overpass_mask,
    )

    # accuracy
    metric_acc = MulticlassAccuracy(num_classes=num_classes)
    metric_acc.update(mask_cs_p_class, mask_cs)
    metric_acc = metric_acc.compute()

    # accuracy by class
    metric_acc_byclass = MulticlassAccuracy(num_classes=num_classes, average=None)
    metric_acc_byclass.update(mask_cs_p_class, mask_cs)
    metric_acc_byclass = metric_acc_byclass.compute()
    metric_acc_byclass = {
        f"{stage}/accuracy_{str(c)}": metric_acc_byclass[c] for c in range(num_classes)
    }

    # diceloss
    metric_func = DiceMetric(
        include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True
    )

    metric_dice = metric_func(cs_p, cs + 1)
    metric_dice_mean = torch.mean(metric_dice)
    # f1
    metric_f1 = MulticlassF1Score(num_classes=num_classes)
    metric_f1.update(mask_cs_p_class, mask_cs)
    metric_f1 = metric_f1.compute()

    # f1 by class
    metric_f1_byclass = MulticlassF1Score(num_classes=num_classes, average=None)
    metric_f1_byclass.update(mask_cs_p_class, mask_cs)
    metric_f1_byclass = metric_f1_byclass.compute()
    metric_f1_byclass = {
        f"{stage}/f1_{str(c)}": metric_f1_byclass[c] for c in range(num_classes)
    }
    return (
        ssi_cs,
        metric_acc,
        metric_f1,
        metric_dice_mean,
        metric_acc_byclass,
        metric_f1_byclass,
    )


def calculate_metrics_per_cloudtype(cs, cs_p, cloud_type, overpass_mask):
    """
    Function to calculate metrics per cloud type.
    cs: true cloudsat profile
    cs_p: predicted cloudsat profile
    cloud_type: cloud type
    """
    results = {}

    cs, _ = get_profiles(cs, cs_p, overpass_mask)
    cloud_type, cs_p = get_profiles(cloud_type, cs_p, overpass_mask)

    assert len(cs) == len(cs_p) == len(cloud_type)
    assert len(cs) == 1, "Expected batch of size 1"

    for cs_i, cs_p_i, cloud_type_i in zip(cs, cs_p, cloud_type):
        for i in range(0, 9):
            # Check for finite values in target & isolate cloud type
            mask_ct = torch.isfinite(cloud_type_i) & (cloud_type_i == i)

            masked_cs = torch.masked_select(cs_i, mask_ct).cpu()
            masked_cs_p = torch.masked_select(cs_p_i, mask_ct).cpu()

            cloudsat_mean = torch.mean(masked_cs).cpu().item()
            cloudsat_std = torch.std(masked_cs).cpu().item()

            prediction_mean = torch.mean(masked_cs_p).cpu().item()
            prediction_std = torch.std(masked_cs_p).cpu().item()

            mse_cs = torch.mean((masked_cs - masked_cs_p) ** 2)
            rmse_cs = torch.sqrt(mse_cs)

            results[f"cloudtype_{i}_mse"] = round(mse_cs.cpu().item(), 4)
            results[f"cloudtype_{i}_rmse"] = round(rmse_cs.cpu().item(), 4)
            results[f"cloudtype_{i}_count"] = int(masked_cs.shape[0])
            results[f"cloudtype_{i}_cloudsat_mean"] = round(cloudsat_mean, 4)
            results[f"cloudtype_{i}_cloudsat_std"] = round(cloudsat_std, 4)
            results[f"cloudtype_{i}_prediction_mean"] = round(prediction_mean, 4)
            results[f"cloudtype_{i}_prediction_std"] = round(prediction_std, 4)

    return results

