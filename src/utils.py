"""
Some functions copied from M3LEO: https://github.com/spaceml-org/M3LEO
"""

from __future__ import annotations

import ast
import functools
import os
import pathlib
from datetime import datetime
from glob import glob

import dotenv
import gcsfs
import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rasterio
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F

from src.config import MSG_WAVELENGTHS

fs = gcsfs.GCSFileSystem()

dotenv.load_dotenv()


def get_activation_fn(activation_str):
    """ Return an activation function callable by name.

        Parameters
        ----------
        activation_str : str. Name of the activation function (e.g. "relu", "elu", "linear", "leaky_relu").

        Returns
        -------
        callable. The corresponding PyTorch activation function.
    """
    if activation_str == "relu":
        return F.relu
    elif activation_str == "elu":
        return F.elu
    elif activation_str == "linear":
        return lambda x: x
    elif activation_str == "leaky_relu":
        return F.leaky_relu

    raise ValueError(f"unknown activation '{activation_str}'")


def find_hydra_run_path(outputs_dir, wandb_runid):
    """
    Find the hydra run path that contains the given wandb run ID.

    Parameters
    ----------
    outputs_dir : str. Root directory of hydra outputs.
    wandb_runid : str. Wandb run ID to search for.

    Returns
    -------
    str. Path to the hydra run directory.
    """

    files = glob(f"**/*{wandb_runid}*", root_dir=outputs_dir, recursive=True)
    if len(files) == 0:
        raise ValueError("no file found")
    files = [f for f in files if "wandb/" in f]
    if len(files) == 0:
        raise ValueError("no wandb log found")

    r = outputs_dir + "/" + files[0].split("/wandb/")[0]
    return r


def load_ckpt_from_hydra_run(
    hydra_run_path: str,
    loading_from_state_dict=True,
    enable_loading_weights: bool = True,
    model: pl.LightningModule = None,
    config=None,
    type="best",
) -> pl.LightningModule:
    """
    Load a checkpoint model from a run's hydra output log.

    Parameters
    ----------
    hydra_run_path : str. File path to the hydra run.
    loading_from_state_dict : bool. If True, load model using state_dict, otherwise use PyTorch Lightning checkpoint.
    enable_loading_weights : bool. If False, model is initialized with random weights.
    model : pl.LightningModule or None. Pre-instantiated model (optional).
    config : OmegaConf or None. Pre-loaded config (optional).
    type : str. Criterion for selecting checkpoint ("best" or "last").

    Returns
    -------
    pl.LightningModule. The loaded PyTorch Lightning module.
    """

    # load config
    if config is None:
        config_file = f"{hydra_run_path}/.hydra/config.yaml"
        if not os.path.isfile(config_file):
            raise ValueError(f"config file {config_file} not found")

        config = OmegaConf.load(config_file)

    # look for checkpoint
    ckpts_paths = [
        f"{hydra_run_path}{'/*'*trailings}/*ckpt" for trailings in range(6)
    ]
    ckpts = functools.reduce(
        lambda lista, elemento: glob(elemento) + lista, ckpts_paths, []
    )
    ckpts = sorted(ckpts)
    print(ckpts)
    if len(ckpts) == 0:
        raise ValueError(f"no checkpoints found in {hydra_run_path}")

    if len(ckpts) > 1:
        print(
            f"there are {len(ckpts)} checkpoints, attempting to use the "
            + type
            + " one"
        )
        try:
            best_ckpt = glob(f"{hydra_run_path}/*/*" + type + "*")[0]
        except IndexError:  # if no "best" model exists
            print(f"could not load best model, attempting last model instead")
            best_ckpt = ckpts[-1]
    else:
        print(f"there is {len(ckpts)} checkpoint")
        best_ckpt = ckpts[-1]

    logger.info(f"loaded checkpoint: {best_ckpt}")

    if model is None:
        logger.info("Creating Model")
        # instantiate model class
        model = hydra.utils.instantiate(config.model)

    # load model
    if enable_loading_weights:
        logger.info("Loading weights from model checkpoint")
        if loading_from_state_dict:
            logger.info("Loading using state_dict")
            checkpoint = torch.load(best_ckpt, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
        else:
            logger.info("Loading using checkpoint from PyTorch Lightning")
            model = model.load_from_checkpoint(best_ckpt)

        logger.info("---------------------------------")
        logger.info(f"model checksum  {print_checksum_of_model(model):.4f}")
        logger.info("---------------------------------")
    else:
        logger.warning(
            "The weights of the pretrained model are not loaded. The model will be initialized with random weights."
        )
        logger.warning("---------------------------------")
        logger.warning(f"model checksum  {print_checksum_of_model(model):.4f}")
        logger.info("---------------------------------")

    return model


def print_checksum_of_model(model):
    """ Compute and return the sum of absolute parameter values as a checksum. """
    return sum(torch.abs(p).sum() for p in model.parameters()).detach().cpu().numpy()


def load_dataloader_from_hydra_run(
    hydra_run_path: str, path_replace=None
):  # -> pl.LightningDataModule:
    """
    Load a dataloader from a run's hydra output log.

    Parameters
    ----------
    hydra_run_path : str. File path to the hydra run.
    path_replace : dict or None. Dictionary of string replacements to apply to the split file path (optional).

    Returns
    -------
    pl.LightningDataModule. The instantiated PyTorch Lightning dataloader.
    """
    # load config
    config_file = f"{hydra_run_path}/.hydra/config.yaml"
    if not os.path.isfile(config_file):
        raise ValueError(f"config file {config_file} not found")

    config = OmegaConf.load(config_file)

    s = config.dataloader.split_file
    if path_replace is not None:
        for k, v in path_replace.items():
            s = s.replace(k, v)
    config.dataloader["split_file"] = s

    # instantiate model class
    dataloader = hydra.utils.instantiate(config.dataloader)
    return dataloader


def check_file_exists(file: str) -> bool:
    """
    Check if a file exists, works for local files and files in a bucket.

    Parameters
    ----------
    file : str. Path to the file (local path or gs:// URI).

    Returns
    -------
    bool. True if the file exists, False otherwise.
    """
    if file.startswith("gs://"):
        return check_file_exists_in_bucket(file)
    else:
        return os.path.isfile(file)


def check_dir_exists(dir: str) -> bool:
    """
    Check if a directory exists, works for local directories and directories in a bucket.

    Parameters
    ----------
    dir : str. Path to the directory (local path or gs:// URI).

    Returns
    -------
    bool. True if the directory exists, False otherwise.
    """
    if dir.startswith("gs://"):
        return check_file_exists_in_bucket(dir)
    else:
        return os.path.isdir(dir)


def check_file_exists_in_bucket(file: str) -> bool:
    """
    Check if a file exists in a GCS bucket.

    Parameters
    ----------
    file : str. GCS URI (gs://) of the file to check.

    Returns
    -------
    bool. True if the file exists, False otherwise.
    """
    return fs.exists(file)


def list_all_files_in_bucket(bucket_name: str) -> list:
    """
    List all files in a GCS bucket.

    Parameters
    ----------
    bucket_name : str. Name of the GCS bucket.

    Returns
    -------
    list. List of file paths in the bucket.
    """
    file_list = fs.ls(bucket_name)
    return file_list


def list_all_files(dir: str) -> list:
    """
    List all files in a directory, works for local directories and GCS buckets.

    Parameters
    ----------
    dir : str. Path to the directory (local path or gs:// URI).

    Returns
    -------
    list. List of file names or paths in the directory.
    """
    if dir.startswith("gs://"):
        return list_all_files_in_bucket(dir)
    else:
        return os.listdir(dir)


def get_dates_from_files(filenames: list[str]) -> list[datetime]:
    """
    Extract dates from a list of filenames.

    Parameters
    ----------
    filenames : list of str. List of filenames to parse dates from.

    Returns
    -------
    list of datetime. List of dates extracted from the filenames.
    """
    # NOTE: Using Cloudsat timestamps for paired patches
    # G16_s20190181545349_e20190181556115_CS_2019018145847_67783_merged_no_flxhr_patch_01

    if "_CS_" in filenames[0]:
        dates = [
            datetime.strptime(filename.split("_CS_")[-1].split("_")[0], "%Y%j%H%M%S")
            for filename in filenames
        ]
    else:
        dates = [
            datetime.strptime(filename.split("_")[0], "%Y%m%d%H%M%S")
            for filename in filenames
        ]
    return dates


def get_split(files: list, split_dict: DictConfig) -> tuple[list, list]:
    """
    Split files based on dataset specification.

    Parameters
    ----------
    files : list. List of file paths to be split.
    split_dict : DictConfig. Dictionary-like object containing the dataset specification (years, months, days).

    Returns
    -------
    list. List of file paths matching the split specification.
    """
    # Extract dates from filenames
    filenames = [file.split("/")[-1] for file in files]
    dates = get_dates_from_files(filenames)
    # Convert to dataframe for easier manipulation
    df = pd.DataFrame({"filename": filenames, "files": files, "date": dates})

    # Check if years, months, and days are specified
    if "years" not in split_dict.keys() or split_dict["years"] is None:
        logger.info("No years specified for split. Using all years.")
        split_dict["years"] = df.date.dt.year.unique().tolist()
    if "months" not in split_dict.keys() or split_dict["months"] is None:
        logger.info("No months specified for split. Using all months.")
        split_dict["months"] = df.date.dt.month.unique().tolist()
    if "days" not in split_dict.keys() or split_dict["days"] is None:
        logger.info("No days specified for split. Using all days.")
        split_dict["days"] = df.date.dt.day.unique().tolist()

    # Determine conditions specified split
    condition = (
        (df.date.dt.year.isin(split_dict["years"]))
        & (df.date.dt.month.isin(split_dict["months"]))
        & (df.date.dt.day.isin(split_dict["days"]))
    )

    # Extract filenames based on conditions
    split_files = df[condition].files.tolist()

    # Check if files are allocated properly
    if len(split_files) == 0:
        raise ValueError("No files found. Check split specification.")

    return split_files


def get_list_filenames(data_path: str = "./", ext: str = "*"):
    """
    Load a list of file names within a directory.

    Parameters
    ----------
    data_path : str. The directory path to search for files (optional).
    ext : str. The file extension to filter the search (optional).

    Returns
    -------
    list of str. A sorted list of file names matching the given extension within the directory.
    """
    pattern = f"*{ext}"
    path = pathlib.Path(data_path)
    file_list = path.rglob(pattern)
    sorted_file_list = sorted(str(file) for file in file_list)
    return sorted_file_list


def load_cloudsat_patch_tiff(patch_filepath: str) -> dict[str, np.ndarray]:
    """Load CloudSat patch from a tiff file.

    Parameters
    ----------
    patch_filepath : str. The path to the tiff file.

    Returns
    -------
    dict of str to np.ndarray. A dictionary containing the data from the tiff file.
    """
    with rasterio.open(patch_filepath) as data:
        # Convert a string to a list
        variable_order_list = ast.literal_eval(data.tags()["variable_order"])

        # Turn data into a dictionary
        data_dict = {}
        for i, var in enumerate(variable_order_list):
            data_dict[var] = data.read(i + 1)

    # turn data_dict['msg_coord_x'] and data_dict['msg_coord_y'] into coordinate tuples
    data_dict["input_image_coords"] = np.dstack(
        (data_dict["input_image_y"], data_dict["input_image_x"])
    )

    # remove msg_coord_y and msg_coord_x from data_dict
    data_dict.pop("input_image_y")
    data_dict.pop("input_image_x")

    return data_dict


def load_msg_tif_file(
    file: str,
    load_wavelengths: bool = True,
    load_coords: bool = True,
    load_cloudmask: bool = True,
    load_overpass_mask: bool = True,
) -> dict[str, any]:
    """Load MSG tif file and export as a dictionary.

    Parameters
    ----------
    file : str. The path to the tif file.
    load_wavelengths : bool. Whether to load the wavelengths in the dictionary (optional).
    load_coords : bool. Whether to load the coordinates in the dictionary (optional).
    load_cloudmask : bool. Whether to load the cloud mask in the dictionary (optional).
    load_overpass_mask : bool. Whether to load the overpass mask in the dictionary (optional).

    Returns
    -------
    dict of str to any. A dictionary containing the data from the tif file.
    """
    data_dict = {}
    # load dataset
    with rasterio.open(file) as src:
        # convert list to a string
        band_names = ast.literal_eval(src.tags()["band_names"])

        num_data_bands = 12

        # extract radiances
        data_dict["data"] = src.read(list(range(1, num_data_bands)))

        if load_wavelengths:
            # extract wavelengths
            band_names_wvl = band_names[: num_data_bands - 1]
            data_dict["wavelengths"] = np.array(
                [MSG_WAVELENGTHS[band] for band in band_names_wvl]
            )

        if load_cloudmask:
            # extract cloud mask
            index_cloud_mask = band_names.index("cloud_mask") + 1
            data_dict["cloud_mask"] = src.read([index_cloud_mask])

        if load_overpass_mask:
            # extract overpass mask
            index_overpass_mask = band_names.index("cloudsat_overpass_mask") + 1
            data_dict["cloudsat_overpass_mask"] = src.read([index_overpass_mask])

        if load_coords:
            # extract coordinates
            index_latitude = band_names.index("latitude") + 1
            index_longitude = band_names.index("longitude") + 1
            data_dict["coords"] = src.read([index_latitude, index_longitude])

    return data_dict


def convert_old_state_dict_keys(state_dict):
    """Convert old state dict keys to match current architecture"""
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Handle backbone architecture nesting changes
        # Old: backbone.class_token -> New: backbone.backbone.class_token
        # Old: backbone.encoder.* -> New: backbone.backbone.encoder.*
        # Old: backbone.decoder.* -> New: backbone.decoder.*
        if key.startswith("backbone.") and not key.startswith("backbone.backbone.") and not key.startswith("backbone.mask_token") and not key.startswith("backbone.decoder"):
            # Add extra backbone layer for encoder components
            new_key = key.replace("backbone.", "backbone.backbone.", 1)

        # Handle decoder architecture changes
        # Old: decoder.out.0.0.conv_block.0.weight
        # New: decoder.out.0.0.blocks.0.0.conv_block.0.weight
        elif (
            key.startswith("decoder.out.")
            and ".conv_block." in key
            and ".blocks." not in key
        ):
            parts = key.split(".")
            # Insert 'blocks.0.0' after decoder.out.X.Y
            if len(parts) >= 4 and parts[0] == "decoder" and parts[1] == "out":
                new_parts = parts[:4] + ["blocks", "0", "0"] + parts[4:]
                new_key = ".".join(new_parts)

        # Handle conv_skip similarly
        elif (
            key.startswith("decoder.out.")
            and ".conv_skip." in key
            and ".blocks." not in key
        ):
            parts = key.split(".")
            if len(parts) >= 4 and parts[0] == "decoder" and parts[1] == "out":
                new_parts = parts[:4] + ["blocks", "0", "0"] + parts[4:]
                new_key = ".".join(new_parts)

        # Handle final layer renaming if needed
        # Old: decoder.out.X.2.weight -> New: decoder.out.X.1.weight
        elif key.startswith("decoder.out.") and key.endswith(".2.weight"):
            new_key = key.replace(".2.weight", ".1.weight")
        elif key.startswith("decoder.out.") and key.endswith(".2.bias"):
            new_key = key.replace(".2.bias", ".1.bias")

        new_state_dict[new_key] = value

    return new_state_dict


def get_checkpoint_path(hr, criterion_best_model="best"):
    """
    Get the checkpoint path based on a selection criterion.

    Parameters
    ----------
    hr : str. Path to the hydra run directory.
    criterion_best_model : str. Criterion string used to select the best checkpoint (optional).

    Returns
    -------
    str. Path to the selected checkpoint file.
    """
    # look for checkpoint
    ckpts_paths = [f"{hr}/*{'/*'*trailings}/*ckpt" for trailings in range(6)]
    ckpts = functools.reduce(
        lambda lista, elemento: glob(elemento) + lista, ckpts_paths, []
    )
    ckpts = sorted(ckpts)
    print(ckpts)
    if len(ckpts) == 0:
        raise ValueError(f"no checkpoints found in {hr}")

    if len(ckpts) > 1:
        print(
            f"there are {len(ckpts)} checkpoints, attempting to use the "
            + criterion_best_model
            + " one"
        )
        try:
            best_ckpt = glob(f"{hr}/*/*" + criterion_best_model + "*")[0]
        except IndexError:  # if no "best" model exists
            print(f"could not load best model, attempting last model instead")
            best_ckpt = ckpts[-1]
    else:
        print(f"there is {len(ckpts)} checkpoint")
        best_ckpt = ckpts[-1]

    return best_ckpt


def load_model_with_fallback(
    hr, loading_from_state_dict, cfgdata, criterion_best_model="best"
):
    """
    Load a model with automatic fallback and state dict key conversion.

    Parameters
    ----------
    hr : str. Path to the hydra run directory.
    loading_from_state_dict : bool. Whether to load from state dict.
    cfgdata : OmegaConf. Hydra config object used to instantiate the model.
    criterion_best_model : str. Criterion for selecting the best checkpoint (optional).

    Returns
    -------
    pl.LightningModule. The loaded model.
    """
    try:
        # Try the standard loading first
        m = load_ckpt_from_hydra_run(
            hr,
            loading_from_state_dict=loading_from_state_dict,
            config=cfgdata,
            type=criterion_best_model,
        )
        logger.info("Successfully loaded model using standard method")
        return m

    except RuntimeError as e:
        if "Missing key(s) in state_dict" in str(
            e
        ) or "Unexpected key(s) in state_dict" in str(e):
            logger.warning(
                "State dict architecture mismatch detected, attempting key conversion..."
            )

            # Load checkpoint manually and convert keys
            ckpt_path = get_checkpoint_path(hr, criterion_best_model)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Convert state dict keys
            converted_state_dict = convert_old_state_dict_keys(checkpoint["state_dict"])

            # Create model using hydra config
            model = hydra.utils.instantiate(cfgdata.model)

            # Load with converted state dict
            missing_keys, unexpected_keys = model.load_state_dict(
                converted_state_dict, strict=False
            )

            if missing_keys:
                logger.warning(
                    f"Missing keys after conversion: {len(missing_keys)} keys"
                )
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys after conversion: {len(unexpected_keys)} keys"
                )

            logger.info("Successfully loaded model with converted state dict")
            logger.info(f"loaded checkpoint: {ckpt_path}")

            return model
        else:
            # Re-raise if it's a different error
            raise e

    except Exception as e:
        # Fallback to the original approach
        logger.warning(f"Standard loading failed ({str(e)}), trying fallback method...")
        model = hydra.utils.instantiate(cfgdata.model)
        logger.info("Successfully instantiated model from config")

        m = load_ckpt_from_hydra_run(
            hr,
            model=model,
            loading_from_state_dict=loading_from_state_dict,
            config=cfgdata,
        )
        logger.info("Successfully loaded checkpoint with fallback method")
        return m

