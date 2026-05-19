from __future__ import annotations

import argparse
import os

import autoroot  # required to import from src
import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import OmegaConf
from progressbar import progressbar as pbar

import wandb
from src import utils
from src.datamodules.utils import convert_to_datetime
from src.transforms.transforms import (
    CloudSatLinearUnormaliseTransform,
    CloudSatLogUnnormaliseTransform,
)
from src.validation_utils import calculate_metrics_per_cloudtype, log_metrics_wandb

# Clear cache
torch.cuda.empty_cache()

# Define variables to extract
VARIABLES = ["Radar_Reflectivity", "IWC", "re", "CloudTypeMask"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wandb_runid", type=str, required=True
)  # The wandb run id of the inference model
parser.add_argument(
    "--wandb_project", type=str, default="inference-2025"
)  # The wandb project to save the inferences to
parser.add_argument(
    "--finetuned_dir", type=str, required=True
)  # The directory where the model is stored
parser.add_argument(
    "--inferences_dir", type=str, required=True
)  # The directory where the inferences are to be saved
parser.add_argument(
    "--data_path", type=str, default=None
)  # [Optional] The path to the data, in case you want to update it from the .hydra/config.yaml
parser.add_argument(
    "--pretrained_dir", type=str, default=None
)  # [Optional] The directory where the model is stored, in case you want to update it from the .hydra/config.yaml
parser.add_argument(
    "--split", type=str, default="test"
)  # The split to make inference on
parser.add_argument(
    "--num_workers", type=int, default=4
)  # The number of workers to use for the dataloader
parser.add_argument(
    "--criterion_best_model", type=str, default="best"
)  # The checkpoint to load
parser.add_argument(
    "--cyclone_prediction", type=bool, default=False
)  # Whether to use cyclone prediction

args = parser.parse_args()

# Extract arguments
wandb_runid = args.wandb_runid
wandb_project = args.wandb_project
data_path = args.data_path

finetuned_dir = args.finetuned_dir
inferences_dir = args.inferences_dir
pretrained_dir = args.pretrained_dir

split = args.split
num_workers = args.num_workers
criterion_best_model = args.criterion_best_model

cyclone_prediction = args.cyclone_prediction

print("\n\n\n------------------------------------------")
print(f"making inference on {wandb_runid} for {split} split")
print("------------------------------------------\n")

# Load the config file
print(f"looking for hydra run {wandb_runid}", flush=True)
hr = utils.find_hydra_run_path(f"{finetuned_dir}", wandb_runid)

print("\n\n---------------------- preparing data", flush=True)
cfgdata = OmegaConf.load(f"{hr}/.hydra/config.yaml")

# Overwrite config arguments
# Overwrite transforms to add cloud type
if "transforms_dict" in cfgdata.dataloader:
    if "goes_cloudsat" in cfgdata.dataloader.transforms_dict:
        cfgdata.dataloader.transforms_dict.goes_cloudsat.cloudsat_variables = VARIABLES
        if data_path is not None:
            cfgdata.dataloader.data_dir_dict.goes_cloudsat = (
                    data_path + "/cloudsat-goes-paired/"
            )
    if "msg_cloudsat" in cfgdata.dataloader.transforms_dict:
        cfgdata.dataloader.transforms_dict.msg_cloudsat.cloudsat_variables = VARIABLES
        if data_path is not None:
            cfgdata.dataloader.data_dir_dict.msg_cloudsat = (
                    data_path + "/cloudsat-msg-paired/"
            )
    if "himawari_cloudsat" in cfgdata.dataloader.transforms_dict:
        cfgdata.dataloader.transforms_dict.himawari_cloudsat.cloudsat_variables = (
            VARIABLES
        )
        if data_path is not None:
            cfgdata.dataloader.data_dir_dict.himawari_cloudsat = (
                    data_path + "/cloudsat-himawari-paired/"
            )

# option for single satellite runs
# transforms are passed without the transforms_dict
if "transforms" in cfgdata.dataloader:
    cfgdata.dataloader.transforms.cloudsat_variables = VARIABLES
    if data_path is not None:
        data_dir = cfgdata.dataloader.data_dir
        data_dir = data_dir[:-1] if data_dir.endswith("/") else data_dir
        folder = data_dir.split("/")[-1]
        cfgdata.dataloader.data_dir = data_path + "/" + folder

# Overwrite to load cloud type mask in dataloader
cfgdata.dataloader.cloudsat_variables = VARIABLES

# Overwrite to add to summary files
cfgdata.dataloader.load_solar = True
cfgdata.dataloader.load_zenith = True

if cyclone_prediction:
    logger.info("Updating config for cyclone predictions...")
    # option for multi-satellite runs
    if "data_dir_dict" in cfgdata.dataloader:
        # drop msg from data_dir_dict
        if "msg_cloudsat" in cfgdata.dataloader.data_dir_dict:
            del cfgdata.dataloader.data_dir_dict["msg_cloudsat"]
        if "msg_cloudsat" in cfgdata.dataloader.transforms_dict:
            del cfgdata.dataloader.transforms_dict["msg_cloudsat"]
        # update parameters for goes & himawari
        satellite_list = []
        if "himawari_cloudsat" in cfgdata.dataloader.data_dir_dict:
            satellite_list.append("himawari_cloudsat")
            if data_path is not None:
                cfgdata.dataloader.data_dir_dict.himawari_cloudsat = (
                        data_path + "/himawari/"
                )
            else:
                cfgdata.dataloader.data_dir_dict.himawari_cloudsat = "/data/2025-esl-3dclouds-extremes-datasets/fine-tuning-reprocessed/cyclones/himawari/"
        if "goes_cloudsat" in cfgdata.dataloader.data_dir_dict:
            satellite_list.append("goes_cloudsat")
            if data_path is not None:
                cfgdata.dataloader.data_dir_dict.goes_cloudsat = data_path + "/goes/"
            else:
                cfgdata.dataloader.data_dir_dict.goes_cloudsat = "/data/2025-esl-3dclouds-extremes-datasets/fine-tuning-reprocessed/cyclones/goes/"
        cfgdata.dataloader.satellites = satellite_list

    # option for single satellite runs
    if "data_dir" in cfgdata.dataloader:
        if "cloudsat-himawari-paired" in cfgdata.dataloader.data_dir:
            if data_path is not None:
                cfgdata.dataloader.data_dir = data_path + "/himawari/"
            else:
                cfgdata.dataloader.data_dir = "/data/2025-esl-3dclouds-extremes-datasets/fine-tuning-reprocessed/cyclones/himawari/"
        if "cloudsat-goes-paired" in cfgdata.dataloader.data_dir:
            if data_path is not None:
                cfgdata.dataloader.data_dir = data_path + "/goes/"
            else:
                cfgdata.dataloader.data_dir = "/data/2025-esl-3dclouds-extremes-datasets/fine-tuning-reprocessed/cyclones/goes/"
        if "cloudsat-msg-paired" in cfgdata.dataloader.data_dir:
            raise ValueError(
                "No cyclones in MSG field of view. Run with different or multi-satellite models instead..."
            )

    if "filter_clear_sky" in cfgdata.dataloader:
        del cfgdata.dataloader["filter_clear_sky"]

# Overwrite dataloader arguments
cfgdata.dataloader["num_workers"] = num_workers
cfgdata.dataloader["batch_size"] = 1  # needs to be one for properly calculating metrics
# Overwrite dataset sizes
cfgdata.limit_train_batches = 1.0
cfgdata.limit_val_batches = 1.0
cfgdata.limit_test_batches = 1.0
# Overwrite wandb arguments
cfgdata.wandb["project"] = wandb_project
# Overwrite model arguments
if "backbone" in cfgdata.model and pretrained_dir is not None:
    pretrained_model = cfgdata.model.backbone["hydra_run_path"]
    if pretrained_model.endswith("/"):
        pretrained_model = pretrained_model[:-1]
    pretrained_model = pretrained_model.split("/")[-1]
    cfgdata.model.backbone["hydra_run_path"] = f"{pretrained_dir}/{pretrained_model}"

# Get the cloudsat variables to predict
cloudsat_variables = cfgdata.model.variable
if (
        type(cloudsat_variables) is not list
        and type(cloudsat_variables) is not omegaconf.listconfig.ListConfig
):
    cloudsat_variables = [cloudsat_variables]
logger.info(f"Cloudsat variables predicted: {cloudsat_variables}")

# set up wandb config
wandb.config = omegaconf.OmegaConf.to_container(
    cfgdata, resolve=True, throw_on_missing=True
)

if cyclone_prediction:
    experiment_name = wandb_runid + f"-cyclones-{criterion_best_model}"
else:
    experiment_name = wandb_runid + f"-{criterion_best_model}"

wandb_logger = WandbLogger(
    name=f"{experiment_name}-{split}",
    project=cfgdata.wandb.project,
    entity=cfgdata.wandb.entity,
    mode=cfgdata.wandb.mode,
)
# Print config to terminal for logging
yaml_str = omegaconf.OmegaConf.to_yaml(cfgdata)
# logger.debug(f"Hydra-config: {yaml_str}")

# Load model
loading_from_state_dict = True

print("\n\n---------------------- loading model", flush=True)

try:
    m = utils.load_ckpt_from_hydra_run(
        hr,
        loading_from_state_dict=loading_from_state_dict,
        config=cfgdata,
        type=criterion_best_model,
    ).cuda()
    logger.info("successfully instantiated model from hydra config.yaml")
    logger.info("successfully loaded ckpt", hr)
except:  # if the run cannot be loaded from the hydra config, use fallback method instead
    logger.info("could not instantiate model from hydra config.yaml, trying fallback method")

    # NOTE: Added to load swinmae models with mismatch in state_dict
    m = utils.load_model_with_fallback(
        hr=hr,
        loading_from_state_dict=loading_from_state_dict,
        cfgdata=cfgdata,
        criterion_best_model=criterion_best_model,
    ).cuda()
    logger.info("successfully loaded model with fallback method")

# Instantiate dataloader
d = hydra.utils.instantiate(cfgdata.dataloader)
d.prepare_data()

traindl = d.train_dataloader()
testdl = d.test_dataloader()
valdl = d.val_dataloader()

if split == "train":
    dl = traindl
elif split == "val":
    dl = valdl
elif split == "test":
    dl = testdl
else:
    raise ValueError(f"invalid split {split}")

# Prepare dictionary for collection of metrics
# Create an empty dictionary for each cloudsat variable
results_dict = {f"{var}": {} for var in cloudsat_variables}

for var in cloudsat_variables:
    results_dict[f"{var}"]["satellite"] = []
    results_dict[f"{var}"]["time"] = []
    results_dict[f"{var}"]["coords"] = []
    results_dict[f"{var}"]["sat_angle"] = []
    results_dict[f"{var}"]["solar_angle"] = []
    results_dict[f"{var}"]["cloudsat_mean"] = []
    results_dict[f"{var}"]["cloudsat_std"] = []
    results_dict[f"{var}"]["cloudsat_max"] = []
    results_dict[f"{var}"]["cloudsat_min"] = []
    results_dict[f"{var}"]["prediction_mean"] = []
    results_dict[f"{var}"]["prediction_std"] = []
    results_dict[f"{var}"]["prediction_max"] = []
    results_dict[f"{var}"]["prediction_min"] = []

    for metric in [
        "mse",
        "rmse",
        "ssim",
        "psnr",
        "mssim",
        "mpsl",
        "dice",
        "bce",
        "cloud_mse",
        "clear_mse",
    ]:
        results_dict[f"{var}"][f"{metric}"] = []

    for i in range(0, 9):
        results_dict[f"{var}"][f"cloudtype_{i}_count"] = []
        results_dict[f"{var}"][f"cloudtype_{i}_mse"] = []
        results_dict[f"{var}"][f"cloudtype_{i}_rmse"] = []
        results_dict[f"{var}"][f"cloudtype_{i}_cloudsat_mean"] = []
        results_dict[f"{var}"][f"cloudtype_{i}_cloudsat_std"] = []
        results_dict[f"{var}"][f"cloudtype_{i}_prediction_mean"] = []
        results_dict[f"{var}"][f"cloudtype_{i}_prediction_std"] = []

num_short_overpass = 0
for i, batch in enumerate(pbar(dl, max_value=len(dl))):
    with torch.no_grad():
        # Extract relevant data from batch
        cloud_type = batch["cloudsat"]["CloudTypeMask"].cuda()
        cloud_type = cloud_type.permute(0, 2, 1)  # [B, H, W] -> [B, L, H]
        overpass_mask = batch["overpass_mask"].cuda()
        binary_overpass_mask = overpass_mask > 0
        # Check that the overpass mask is longer than 15 pixels
        if torch.sum(binary_overpass_mask) < 15:
            num_short_overpass = num_short_overpass + 1
            logger.info("short overpass, less than 15 pixels. Skipping...")
            continue

        # Move everything to cuda
        for key in set(list(batch.keys())).intersection(
                ["data", "overpass_mask", "coords", "time", "sat_angle", "solar_angle"]
        ):
            batch[key] = batch[key].cuda()
        for var in cloudsat_variables:
            batch["cloudsat"][var] = batch["cloudsat"][var].cuda()
        cs_pred = m.forward(batch)
        if len(cs_pred.shape) == 4:
            cs_pred = cs_pred.unsqueeze(
                1
            )  # Add an extra dimension if only one variable was predicted

        for j, var in enumerate(cloudsat_variables):
            cs = batch["cloudsat"][var]
            cs = cs.permute(0, 2, 1)  # [B, H, W] -> [B, L, H]

            results_dict[f"{var}"]["satellite"].append(batch["satellite"][0])
            results_dict[f"{var}"]["time"].append(
                convert_to_datetime(batch["time"][0].cpu().numpy())
            )
            results_dict[f"{var}"]["coords"].append(
                [
                    round(np.nanmedian(batch["coords"][0, 0, ...].cpu().numpy()), 4),
                    round(np.nanmedian(batch["coords"][0, 1, ...].cpu().numpy()), 4),
                ]
            )
            results_dict[f"{var}"]["sat_angle"].append(
                [
                    round(np.nanmedian(batch["sat_angle"][0, 0, ...].cpu().numpy()), 4),
                    round(np.nanmedian(batch["sat_angle"][0, 1, ...].cpu().numpy()), 4),
                ]
            )
            results_dict[f"{var}"]["solar_angle"].append(
                [
                    round(
                        np.nanmedian(batch["solar_angle"][0, 0, ...].cpu().numpy()), 4
                    ),
                    round(
                        np.nanmedian(batch["solar_angle"][0, 1, ...].cpu().numpy()), 4
                    ),
                ]
            )

            # Unnormalize the predictions
            if var == "Radar_Reflectivity":
                unnormalize_transform = CloudSatLinearUnormaliseTransform(
                    min=-30, max=20
                )
                cs_pred_unnorm = unnormalize_transform(cs_pred[:, j, ...])
                cs_unnorm = unnormalize_transform(cs)
            elif var == "IWC":
                unnormalize_transform = CloudSatLogUnnormaliseTransform(
                    min=1e-5, max=10
                )
                cs_pred_unnorm = unnormalize_transform(cs_pred[:, j, ...])
                cs_unnorm = unnormalize_transform(cs)
            elif var == "re":
                unnormalize_transform = CloudSatLinearUnormaliseTransform(
                    min=0, max=160
                )
                cs_pred_unnorm = unnormalize_transform(cs_pred[:, j, ...])
                cs_unnorm = unnormalize_transform(cs)
            else:
                raise ValueError(f"No unnormalization implemented for variable {var}")

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
                cs=cs_unnorm,
                cs_p=cs_pred_unnorm,
                overpass_mask=overpass_mask,
                stage=f"{split}/{var}",
                experiment=wandb_logger.experiment,
            )

            for metric in [
                "mse",
                "rmse",
                "ssim",
                "psnr",
                "mssim",
                "mpsl",
                "dice",
                "bce",
                "cloud_mse",
                "clear_mse",
            ]:
                results_dict[f"{var}"][f"{metric}"].append(
                    round(float(locals()[f"{metric}"]), 4)
                )

            results_dict[f"{var}"]["cloudsat_mean"].append(
                round(np.nanmean(cs_unnorm.cpu().numpy()), 4)
            )
            results_dict[f"{var}"]["cloudsat_std"].append(
                round(np.nanstd(cs_unnorm.cpu().numpy()), 4)
            )
            results_dict[f"{var}"]["cloudsat_max"].append(
                round(np.nanmax(cs_unnorm.cpu().numpy()), 4)
            )
            results_dict[f"{var}"]["cloudsat_min"].append(
                round(np.nanmin(cs_unnorm.cpu().numpy()), 4)
            )
            results_dict[f"{var}"]["prediction_mean"].append(
                round(np.nanmean(cs_pred_unnorm.cpu().numpy()), 4)
            )
            results_dict[f"{var}"]["prediction_std"].append(
                round(np.nanstd(cs_pred_unnorm.cpu().numpy()), 4)
            )
            results_dict[f"{var}"]["prediction_max"].append(
                round(np.nanmax(cs_pred_unnorm.cpu().numpy()), 4)
            )
            results_dict[f"{var}"]["prediction_min"].append(
                round(np.nanmin(cs_pred_unnorm.cpu().numpy()), 4)
            )

            cloudtype_dict = calculate_metrics_per_cloudtype(
                cs=cs_unnorm,
                cs_p=cs_pred_unnorm,
                cloud_type=cloud_type,
                overpass_mask=overpass_mask,
            )

            for key, value in cloudtype_dict.items():
                results_dict[f"{var}"][key].append(round(float(value), 4))

    if i == (len(dl) - 1):
        logger.info(f"Finished inference on {split} split for {wandb_runid} run.")
        break

logger.info(f"Number of discarded examples: {num_short_overpass}")
logger.info(f"Saving inference files ...")

os.makedirs(inferences_dir, exist_ok=True)
for var in cloudsat_variables:
    inferences_file = f"{inferences_dir}/{experiment_name}-{split}-{var}.csv"
    var_dict = results_dict[f"{var}"]
    df = pd.DataFrame(var_dict)
    # temporarily safe to convert format of lists
    df.to_csv("tmp.csv")
    df = pd.read_csv("tmp.csv")
    df_no_duplicates = (
        df.drop_duplicates()
    )  # remove duplicates from restarted dataloaders.
    df_no_duplicates = df_no_duplicates.sort_values(by="satellite")  # sort by satellite
    df_no_duplicates = df_no_duplicates.reset_index(drop=True)  # reset index
    df_no_duplicates.to_csv(inferences_file, index=False)
    os.remove("tmp.csv")

    logger.info(f"Saved inference file for variable {var}: {inferences_file}")
