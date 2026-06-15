from __future__ import annotations

import ast
from typing import Callable

import autoroot  # required for imports from src
import numpy as np
import xarray as xr
from loguru import logger
from torch.utils.data import Dataset

from src.datamodules.constants import HIMAWARI_WAVELENGTHS
from src.datamodules.utils import (
    convert_to_datetime,
    get_satellite_viewing_angles,
    get_sza_and_azi,
    pad_track_with_nan,
    parse_datatree_time,
    standardize_patch_size,
)


def _process_himawari_datatree(
    file: str,
    load_zenith: bool,
    load_solar: bool,
    load_overpass_mask: bool,
    cloudsat_variables: list[str],
):
    with xr.open_datatree(file, decode_timedelta=True) as goes_datatree:
        time = parse_datatree_time(file)

        ds = goes_datatree.geo_patch.to_dataset()

        # define an empty dictionary
        data_dict = {}

        # add time
        data_dict["time"] = time

        # extract data
        if "data" in ds.data_vars:
            data_dict["data"] = ds.data.values.astype(np.float32)
        else:
            data_dict["data"] = (
                ds[list(HIMAWARI_WAVELENGTHS.keys())]
                .to_array()
                .values.astype(np.float32)
            )

        # extract coordinates
        # calculate latitude and longitude coordinates
        # Fix lat/lon encoding bug
        lat_offset = (
            (ds.latitude.encoding["scale_factor"] + ds.latitude.encoding["add_offset"])
            * 2
            if ds.latitude.encoding["add_offset"] > 0
            else 0
        )
        lon_offset = (
            (
                ds.longitude.encoding["scale_factor"]
                + ds.longitude.encoding["add_offset"]
            )
            * 2
            if ds.longitude.encoding["add_offset"] > 0
            else 0
        )
        latitudes = (ds.latitude - lat_offset).fillna(
            ds.latitude.encoding["add_offset"]
        )
        longitudes = (ds.longitude - lon_offset).fillna(
            ds.longitude.encoding["add_offset"]
        )

        # calculate latitude and longitude coordinates
        data_dict["coords"] = np.stack([latitudes.values, longitudes.values], axis=0)

        # add band names and wavelengths
        data_dict["band_names"] = list(HIMAWARI_WAVELENGTHS.keys())
        data_dict["wavelengths"] = [
            val.get("center_wavelength") for val in HIMAWARI_WAVELENGTHS.values()
        ]
        data_dict["sensor_info"] = HIMAWARI_WAVELENGTHS

        if load_overpass_mask:
            data_dict["overpass_mask"] = ds.cloudsat_overpass_mask.values

        # calculate the zenith angle
        if load_zenith:
            if "sat_angle" in ds.data_vars:
                data_dict["sat_angle"] = ds.sat_angle.values
            else:
                zenith, azimuth = get_satellite_viewing_angles(
                    lat=data_dict["coords"][0],
                    lon=data_dict["coords"][1],
                    sat_lat=ast.literal_eval(ds.B01.orbital_parameters)[
                        "projection_latitude"
                    ],
                    sat_lon=ast.literal_eval(ds.B01.orbital_parameters)[
                        "projection_longitude"
                    ],
                    sat_alt=ast.literal_eval(ds.B01.orbital_parameters)[
                        "projection_altitude"
                    ]
                    / 1e3,  # convert to km
                )
                data_dict["sat_angle"] = np.stack([zenith, azimuth], axis=0)
        if load_solar:
            if "solar_angle" in ds.data_vars:
                data_dict["solar_angle"] = ds.solar_angle.values
            else:
                time = convert_to_datetime(time)
                zenith, azimuth = get_sza_and_azi(
                    date=time, lat=data_dict["coords"][0], lon=data_dict["coords"][1]
                )
                data_dict["solar_angle"] = np.stack([zenith, azimuth], axis=0)

        # pad CloudSat track with Nans
        data_dict["cloudsat"] = pad_track_with_nan(
            datatree=goes_datatree,
            cloudsat_variables=cloudsat_variables,
        )

    return data_dict


class CloudsatHIMAWARIDataset(Dataset):
    """
    Class to load HIMAWARI and CloudSat data.
    """

    def __init__(
        self,
        data_filenames: list[str],
        transforms: Callable | None = None,
        load_overpass_mask: bool = True,
        load_zenith: bool = True,
        load_solar: bool = True,
        cloudsat_variables: list[str] = ["Radar_Reflectivity"],
    ):
        self.data_filenames = data_filenames
        self.transforms = transforms
        self.load_overpass_mask = load_overpass_mask
        self.load_zenith = load_zenith
        self.load_solar = load_solar
        self.cloudsat_variables = cloudsat_variables
        self.max_attempts = 20  # Maximum number of attempts to load valid data

    def setup(self, stage):
        pass

    def prepare_data(self):
        pass

    def __getitem__(self, ind):  # can output array or dict depending on transforms
        attempts_remaining = self.max_attempts  # Local copy for this call
        while attempts_remaining > 0:
            try:
                file = self.data_filenames[ind]
                data_dict = _process_himawari_datatree(
                    file=file,
                    load_overpass_mask=self.load_overpass_mask,
                    load_zenith=self.load_zenith,
                    load_solar=self.load_solar,
                    cloudsat_variables=self.cloudsat_variables,
                )
                break  # If the file is successfully loaded, break the loop

            except Exception as e:
                # Log the error with remaining attempts
                logger.warning(
                    f"Error loading {self.data_filenames[ind]}. "
                    f"Attempts remaining: {attempts_remaining}. "
                    f"Error: {e}"
                )

                # If we have no attempts left, raise an error
                if attempts_remaining <= 0:
                    raise ValueError(
                        f"Could not load valid file after {self.max_attempts} attempts. "
                        f"Last error: {e}"
                    )

                # If there is an error, try to load another file
                ind = np.random.randint(0, len(self.data_filenames))
                attempts_remaining -= 1
                continue

        # apply transforms to data dictionary
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)

        # Standardize patch sizes to ensure consistent dimensions
        data_dict = standardize_patch_size(data_dict, target_size=(256, 256))

        # add center coordinates and angles after transforms to calculate from small patch
        data_dict["center_coords"] = np.nanmedian(data_dict["coords"], axis=[1, 2])
        if self.load_zenith:
            data_dict["center_sat_angle"] = np.nanmedian(
                data_dict["sat_angle"], axis=[1, 2]
            )
        if self.load_solar:
            data_dict["center_solar_angle"] = np.nanmedian(
                data_dict["solar_angle"], axis=[1, 2]
            )

        data_dict["satellite"] = "himawari_cloudsat"

        return data_dict

    def __len__(self):
        return len(self.data_filenames)
