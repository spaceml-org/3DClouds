from __future__ import annotations

import ast
import gc
from typing import Callable

import autoroot  # required for imports from src
import numpy as np
import xarray as xr
from loguru import logger
from torch.utils.data import Dataset

from src.datamodules.constants import HIMAWARI_WAVELENGTHS
from src.datamodules.utils import (
    CropDataset,
    convert_to_datetime,
    get_satellite_viewing_angles,
    get_sza_and_azi,
    parse_time,
)


def load_himawari_file(
    file: str,
    load_zenith: bool = True,
    load_solar: bool = True,
    load_overpass_mask: bool = False,  # Needs to be false by default, because pre-training patches don't have overpass mask
    patch_size: list | None = None,  # Whether to crop the data to a smaller patch size (e.g. [128, 128] for pre-training)
    center_crop: bool = False,  # If True, will crop to the center of the image
    radius: int = 0,  # Radius for cropping, if center_crop is True
):
    if not file.endswith(".nc"):
        raise NotImplementedError("Unsupported file format.")

    # define an empty dictionary
    data_dict = {}
    # open file
    with xr.open_dataset(file) as ds:

        if patch_size is not None:
            crop_ds = CropDataset(
                patch_size=patch_size,
                center_crop=center_crop,  # If True, will crop to the center of the image
                radius=radius,
            )
            ds = crop_ds(ds)

        # extract data
        if "data" in ds.data_vars:
            data_dict["data"] = ds.data.values.astype(np.float32)
        else:
            data_dict["data"] = ds[list(HIMAWARI_WAVELENGTHS.keys())].to_array().values.astype(np.float32)

        # extract coordinates
        lat_offset = (
            (ds.latitude.encoding["scale_factor"] + ds.latitude.encoding["add_offset"])*2
            if ds.latitude.encoding["add_offset"] > 0
            else 0
        )
        lon_offset = (
            (ds.longitude.encoding["scale_factor"] + ds.longitude.encoding["add_offset"])*2
            if ds.longitude.encoding["add_offset"] > 0
            else 0
        )
        latitudes = (ds.latitude - lat_offset).fillna(ds.latitude.encoding["add_offset"])
        longitudes = (ds.longitude - lon_offset).fillna(ds.longitude.encoding["add_offset"])
        
        data_dict["coords"] = np.stack([latitudes.values, longitudes.values], axis=0)

        # get time from file name
        data_dict["time"] = parse_time(file)

        # add band names and wavelengths
        data_dict["band_names"] = list(HIMAWARI_WAVELENGTHS.keys())
        data_dict["wavelengths"] = [
            val.get("center_wavelength") for val in HIMAWARI_WAVELENGTHS.values()
        ]
        data_dict["sensor_info"] = HIMAWARI_WAVELENGTHS

        if load_overpass_mask:
            raise NotImplementedError

        # calculate the zenith angle
        if load_zenith:
            if "sat_angle" in ds.data_vars:
                data_dict["sat_angle"] = ds.sat_angle.values
            else:
                zenith, azimuth = get_satellite_viewing_angles(
                    lat=ds["latitude"].values,
                    lon=ds["longitude"].values,
                    sat_lat=ast.literal_eval(ds.B01.orbital_parameters)["projection_latitude"],
                    sat_lon=ast.literal_eval(ds.B01.orbital_parameters)["projection_longitude"],
                    sat_alt=ast.literal_eval(ds.B01.orbital_parameters)["projection_altitude"]
                    / 1e3,  # convert to km
                )
                data_dict["sat_angle"] = np.stack([zenith, azimuth], axis=0)
        if load_solar:
            if "solar_angle" in ds.data_vars:
                data_dict["solar_angle"] = ds.solar_angle.values
            else:
                time = convert_to_datetime(data_dict["time"])
                zenith, azimuth = get_sza_and_azi(date=time, lat=data_dict["coords"][0], lon=data_dict["coords"][1])
                data_dict["solar_angle"] = np.stack([zenith, azimuth], axis=0)
    
    return data_dict


class HIMAWARIDataset(Dataset):
    """
    Class to load HIMAWARI data.
    """

    def __init__(
        self,
        data_filenames: list[str],
        transforms: Callable | None = None,
        return_overpass_mask: bool = False,
        load_zenith: bool = True,
        load_solar: bool = True,
        patch_size: list[int] | None = None,  # Patch size for cropping the data
        center_crop: bool = False,  # If True, will crop to the center of the image
        radius: int = 0,  # Radius for cropping, if center_crop is True
    ):
        self.data_filenames = data_filenames
        self.transforms = transforms
        self.return_overpass_mask = return_overpass_mask
        self.patch_size = patch_size
        self.load_zenith = load_zenith
        self.load_solar = load_solar
        self.patch_size = patch_size
        self.max_attempts = 20  # Maximum number of attempts to load valid data
        self.center_crop = center_crop  # If True, will crop to the center of the image
        self.radius = radius  # Radius for cropping, if center_crop is True

    def setup(self, stage):
        pass

    def prepare_data(self):
        pass

    def __getitem__(self, ind):
        attempts_remaining = self.max_attempts  # Local copy for this call
        while attempts_remaining > 0:
            try:
                # Check that there are no errors loading the file
                file_path = self.data_filenames[ind]
                data_dict = load_himawari_file(
                    file=file_path,
                    load_zenith=self.load_zenith,
                    load_solar=self.load_solar,
                    load_overpass_mask=self.return_overpass_mask,
                    patch_size=self.patch_size,
                    center_crop=self.center_crop,
                    radius=self.radius,
                )
                break  # If the file is successfully loaded, break the loop

            except Exception as e:
                # If we have no attempts left, raise an error
                if attempts_remaining <= 0:
                    raise ValueError(
                        f"Could not load valid file after {self.max_attempts} attempts. "
                        f"Error: {e}"
                    )

                # KeyErrors can arise if any of the variables are missing
                logger.warning(
                    f"Error loading {self.data_filenames[ind]}. "
                    f"Attempting with other files. "
                    f"Error: {e}"
                )

                # If there is an error, try to load another file
                ind = np.random.randint(0, len(self.data_filenames))
                attempts_remaining -= 1
                continue

        # Apply transformations
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        # Add center coordinates and angles
        data_dict["center_coords"] = np.nanmedian(data_dict["coords"], axis=[1, 2])
        if self.load_zenith:
            data_dict["center_sat_angle"] = np.nanmedian(
                data_dict["sat_angle"], axis=[1, 2]
            )
        if self.load_solar:
            data_dict["center_solar_angle"] = np.nanmedian(
                data_dict["solar_angle"], axis=[1, 2]
            )

        data_dict["satellite"] = "himawari"

        return data_dict

    def __len__(self):
        return len(self.data_filenames)
