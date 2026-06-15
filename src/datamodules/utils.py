from __future__ import annotations

import glob
import os
import random
from datetime import datetime, timedelta
from random import randint
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger
from pyproj import Proj

from src.datamodules.constants import CLOUDSAT_NAN_FILL_VALUES


def standardize_patch_size(data_dict: dict, target_size: tuple = (256, 256)) -> dict:
    target_h, target_w = target_size

    # Define padding strategies for different data types
    padding_config = {
        "spatial_keys": {
            "keys": [
                "data",
                "coords",
                "sat_angle",
                "solar_angle",
                "coords_rad",
                "sat_angle_rad",
                "solar_angle_rad",
                "time_2d",
            ],
            "mode": "edge",
            "constant_values": None,
        },
        "overpass_mask": {
            "keys": ["overpass_mask"],
            "mode": "constant",
            "constant_values": 0,
        },
    }

    # Process all non-CloudSat data with unified logic
    for config_name, config in padding_config.items():
        for key in config["keys"]:
            if key not in data_dict or len(data_dict[key].shape) < 2:
                continue

            arr = data_dict[key]
            current_h, current_w = arr.shape[-2:]

            if current_h == target_h and current_w == target_w:
                continue

            # Crop if needed
            if current_h > target_h or current_w > target_w:
                crop_h = min(current_h, target_h)
                crop_w = min(current_w, target_w)
                if len(arr.shape) == 3:
                    arr = arr[:, :crop_h, :crop_w]
                else:
                    arr = arr[:crop_h, :crop_w]

            # Pad if needed
            if current_h < target_h or current_w < target_w:
                pad_h = target_h - current_h
                pad_w = target_w - current_w

                if len(arr.shape) == 3:
                    pad_width = ((0, 0), (0, pad_h), (0, pad_w))
                else:
                    pad_width = ((0, pad_h), (0, pad_w))

                pad_kwargs = {"mode": config["mode"]}
                if config["constant_values"] is not None:
                    pad_kwargs["constant_values"] = config["constant_values"]

                arr = np.pad(arr, pad_width, **pad_kwargs)

            data_dict[key] = arr

    return data_dict


def parse_time(data_filename: str):
    """
    Parse the timestep from the given data filename.

    Args:
        data_filename (str): The filename of the data.

    Returns:
        numpy.ndarray: An array containing the year, month, day, hour, and minute of the timestep.

    """
    filename = data_filename.split("/")[-1]
    filename_bits = filename.split("_")
    datetime_str = filename_bits[0]

    datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
    fraction_of_year = np.clip(int(datetime_obj.strftime("%j")) / 365, 0, 1)
    fraction_of_day = np.clip(
        datetime_obj.hour / 24
        + datetime_obj.minute / 24 / 60
        + datetime_obj.second / 24 / 60 / 60,
        0,
        1,
    )
    return np.array(
        [
            int(datetime_obj.year),
            int(datetime_obj.month),
            int(datetime_obj.day),
            int(datetime_obj.hour),
            int(datetime_obj.minute),
            int(datetime_obj.second),
            fraction_of_year,
            fraction_of_day,
        ]
    )


def parse_datatree_time(data_filename: str):
    """
    Parse the timestep from the given data filename.

    Args:
        data_filename (str): The filename of the data.

    Returns:
        numpy.ndarray: An array containing the year, month, day, hour, and minute of the timestep.

    """
    filename = data_filename.split("/")[-1]
    filename_bits = filename.split("_")

    # GOES: G16_sYYYYMMDDHHMMSSs_...
    # Himawari: H08_YYYYMMDD_HHMM_...
    # MSG: MSG1_20060612225740_...

    if "G16" in filename:
        datetime_str = filename_bits[1][
            1:-1
        ]  # remove the s at the start and the first decimal second at the end
        datetime_obj = datetime.strptime(
            datetime_str, "%Y%j%H%M%S"
        )  # %j means day of the year
    elif "H08" in filename:
        datetime_str = (
            filename_bits[1] + filename_bits[2] + "00"
        )  # no seconds in filename, set to 00
        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
    elif "MSG" in filename:
        datetime_str = filename_bits[1]
        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
    elif "FLDK" in filename:  # TODO check if this is correct for FLDK
        datetime_str = filename_bits[0] + filename_bits[1] + "00"
        datetime_obj = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")
    else:
        raise NotImplementedError("filename not recognised")

    fraction_of_year = np.clip(int(datetime_obj.strftime("%j")) / 365, 0, 1)
    fraction_of_day = np.clip(
        datetime_obj.hour / 24
        + datetime_obj.minute / 24 / 60
        + datetime_obj.second / 24 / 60 / 60,
        0,
        1,
    )
    return np.array(
        [
            int(datetime_obj.year),
            int(datetime_obj.month),
            int(datetime_obj.day),
            int(datetime_obj.hour),
            int(datetime_obj.minute),
            int(datetime_obj.second),
            fraction_of_year,
            fraction_of_day,
        ]
    )


def convert_to_datetime(time: np.array):
    """
    Function to convert an array to a datetime object.
    Time is structured as [YYYY, MM, DD, HH, MM, SS]
    """

    time_dt = datetime(
        year=int(time[0]),
        month=int(time[1]),
        day=int(time[2]),
        hour=int(time[3]),
        minute=int(time[4]),
        second=int(time[5]),
    )

    return time_dt


def parse_center_coords(x: dict):
    """
    Get center coordinates for a given patch.

    Args:
        x (dict): A dictionary containing the data.

    Returns:
        numpy.ndarray: Array containing lat and lon of the patch center.
    """
    lats, lons = x["coords"][0], x["coords"][1]
    # Filter out NaN and Inf values
    valid_lats = lats[np.isfinite(lats)]
    valid_lons = lons[np.isfinite(lons)]
    # Calculate mean of valid values
    center_lat = np.mean(valid_lats)
    if np.isnan(center_lat) or np.isinf(center_lat):
        # If all values are invalid, set to 90 (i.e. out of MSG FOV)
        center_lat = 90
    center_lon = np.mean(valid_lons)
    if np.isnan(center_lon) or np.isinf(center_lon):
        # If all values are invalid, set to 180 (i.e. out of MSG FOV)
        center_lon = 180
    center_coords = np.array([center_lat, center_lon])
    return center_coords


def filter_files(filenames: list, filter_clear_sky: dict, satellite: str):
    """
    Filter files based on the clear sky condition.

    Args:
        filenames (list): List of filenames to filter.
        filter_clear_sky (dict): Dictionary containing the filter conditions.
            -- path (str): Path to the summary file.
            -- key (str): Key to filter by.
            -- cutoff (float): Cutoff value for filtering.

    Returns:
        list: List of filtered filenames.
    """

    filter_clear_sky = filter_clear_sky.copy()
    if "path" not in filter_clear_sky.keys():
        raise KeyError("Key 'path' must be provided in filter_clear_sky dictionary.")
    path = filter_clear_sky.pop("path")

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)

    df["sensor"] = df["sensor"].str.lower()
    df_sat = df[df["sensor"] == satellite.lower()].reset_index(drop=True)

    filters = []
    for key, [operator, value] in filter_clear_sky.items():
        if key not in df_sat.columns:
            raise KeyError(f"Key '{key}' not found in DataFrame columns.")

        match operator:
            case ">":
                filters.append((df_sat[key] > value).tolist())
            case ">=":
                filters.append((df_sat[key] >= value).tolist())
            case "<":
                filters.append((df_sat[key] < value).tolist())
            case "<=":
                filters.append((df_sat[key] <= value).tolist())
            case "==":
                filters.append((df_sat[key] == value).tolist())
            case "!=":
                filters.append((df_sat[key] != value).tolist())
            case _:
                raise ValueError(f"Operator {operator} not valid comparison")

    filenames_to_select = df_sat[np.logical_and.reduce(filters)]["file"].tolist()

    base_path = os.path.commonpath(filenames)
    filenames_to_check = [os.path.basename(f) for f in filenames]

    overlap = set(filenames_to_check) & set(filenames_to_select)
    overlap_filenames = [os.path.join(base_path, f) for f in overlap]

    return overlap_filenames


def filter_filenames_time(file: str, finetuning: bool = False, cutoff: int = 2.05):
    """
    Function to load timesteps before and after a given file.
    Loads the same patch on the same day.

    Args:
        file (str): The filename of the data.
        finetuning (bool): If finetuning, the patches have different names
        cutoff (int): The number of hours to load before and after the given file.

    Returns:
        list[str], list[str]: List of filenames before and after the given file
    """
    folders = file.split("/")[:-1]
    tif_path = "/".join(folders)
    tif_time = file.split("/")[-1].split("_")[0]

    tif_datetime = datetime.strptime(tif_time, "%Y%m%d%H%M%S")

    delta = timedelta(hours=cutoff)
    tif_datetime_before = tif_datetime - delta
    tif_datetime_after = tif_datetime + delta

    patch_num = file.split("/")[-1].split("_")[-1].split(".")[0]
    ext = file.split(".")[-1]

    if finetuning:
        # Fine-tuning patches need to references in relation to cloudsat patches
        cloudsat_num = file.split("/")[-1].split("_")[1]
        # NOTE: Added advanced filtering of patch names
        all_files = glob.glob(
            tif_path + f"/{tif_time}_{cloudsat_num}_*_msg_ts_patch_{patch_num}.{ext}"
        )

    else:
        all_files = glob.glob(tif_path + f"/*_{patch_num}.{ext}")

    # Filter files within the datetime range
    tif_files_before = []
    tif_files_after = []
    for f in all_files:
        # Extract datetime from filename
        fname_datetime_str = f.split("/")[-1].split("_")[2]  # NOTE: 2 ????
        fname_datetime = datetime.strptime(fname_datetime_str, "%Y%m%d%H%M%S")

        # Check if the file's datetime is within the range
        if tif_datetime_before <= fname_datetime <= tif_datetime:
            tif_files_before.append(f)
        if tif_datetime_after >= fname_datetime >= tif_datetime:
            tif_files_after.append(f)

    # NOTE: Added sorting of filenames and deleting of duplicates
    # Remove duplicates (if they exist)
    tif_files_before = list(sorted(set(tif_files_before)))
    tif_files_after = list(sorted(set(tif_files_after)))

    return tif_files_before, tif_files_after


def pick_random_files(file_list: list[str], central_file: str, num_files: int):
    """
    Function to pick random files from a list of files.

    Args:
        file_list (list[str]): List of filenames to pick from.
        central_file (str): The central file. file_list will be extended with central file if needed.
        num_files (int): The number of files to select.
    """
    while len(file_list) < num_files:
        file_list.append(central_file)

    return sorted(random.sample(file_list, num_files))


def pick_ordered_files(
    file_list: list[str], central_file: str, num_files: int, order: str
):
    """
    Function to pick files from a list in order.

    Args:
        file_list (list[str]): List of filenames to pick from.
        central_file (str): The central file. file_list will be extended with central file if needed.
        num_files (int): The number of files to select.
        order (str): The order to pick files. Either "start" or "end".
    """
    logger.info(
        "Length of timeseries shorter than number of timesteps requested. Appending central time."
    ) if len(file_list) < num_files else None
    while len(file_list) < num_files:
        file_list.append(central_file)

    # Order files by time
    file_list = sorted(file_list)

    if order == "start":
        return file_list[:num_files]
    elif order == "end":
        return file_list[-num_files:]


def get_satellite_viewing_angles(
    lat: np.ndarray,
    lon: np.ndarray,
    sat_lat: float,
    sat_lon: float,
    sat_alt: float,  # in km
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate satellite zenith and azimuth angles.

    Satellite zenith angle measures the angle from vertical that an observation
    is made at the surface. 0 means that the satellite is directly overhead. 90
    means that the surface point is on the horizon of the satellite view.

    Satellite azimuth angle measures the angle from North from the surface point
    to the satellite, measured clockwise. 0 is due N, 90 is E, 180 is S and 270
    is W.

    Parameters
    ----------
    lat : np.ndarray
        latitudes of surface point in degrees
    lon : np.ndarray
        longitudes of surface point in degrees
    sat_lat : float, optional
        latitude of sub-satellite point in degrees, by default 0
    sat_lon : float, optional
        longitude of sub-satellite point in degrees, by default 0
    sat_alt : float, optional
        altitude of satellite in km, by default 35_793 (geostationary orbit
        height over average earth radius)

    Returns
    -------
    tuple[float, float]
        satellite zenith and azimuth angles in degrees
    """
    # TODO test for inf / nan coordinates
    # Approximate spherical Earth so use radius of 6,371 km
    Re = 6_371
    Rgeo = sat_alt + Re

    # Caclulate the beta angle
    cos_beta = np.cos(np.radians(lat - sat_lat)) * np.cos(np.radians(lon - sat_lon))
    sin_beta = np.sin(np.arccos(cos_beta))

    # Calculate satellite zenith angle
    geo_dist = (
        Rgeo**2 + Re**2 - 2 * Rgeo * Re * cos_beta
    ) ** 0.5  # distance from surface to satellite
    sin_theta = (Rgeo * sin_beta) / geo_dist
    zenith_angle = np.degrees(np.arcsin(sin_theta))

    # Find where satellite-surface path intersects the earth and make these > 90
    zenith_angle = np.where(
        geo_dist**2 < (Rgeo**2 - Re**2), zenith_angle, 180 - zenith_angle
    )

    # Calculate satellite azimuthal angle
    x_sat = np.cos(np.radians(lat - sat_lat)) * np.sin(np.radians(lon - sat_lon))
    y_sat = np.sin(np.radians(lat - sat_lat))
    azimuth_angle = np.where(
        np.isfinite(x_sat), np.degrees(np.arctan2(x_sat, y_sat)) % 360, np.nan
    )

    return zenith_angle, azimuth_angle


def get_sza_and_azi(
    date: datetime, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get the solar zenith angle at a specific time/lat/lon

    Parameters
    ----------
    date : datetime | array of datetime like
        Dates of the points
    lat : np.ndarray
        Latitudes
    lon : np.ndarray
        Longitudes

    Returns
    -------
    sza: np.ndarray
        The solar zenith angle in degrees, where 0 is directly above, 90 is on
        the horizon and 180 is directly below
    saa: np.ndarray
        The solar azimuth angle in degrees, clockwise from North
    """
    # TODO test for inf / nan coordinates
    try:
        date = pd.DatetimeIndex(date)
    except TypeError:
        date = pd.DatetimeIndex([date])
    day_of_year = date.dayofyear.to_numpy()
    hour_of_day = (date.hour + date.minute / 60 + date.second / 60 / 60).to_numpy()

    # calculate approx time equation as angle for 365 day year
    equation_of_time_approx = 2.0 * np.pi * day_of_year / 365.0

    # calculate the solar declination for the given day
    # the declination varies due to the fact that the earth rotation axis
    # is not perpendicular to the ecliptic plane
    solar_declination = (
        0.006918
        - 0.399912 * np.cos(equation_of_time_approx)
        - 0.006758 * np.cos(2.0 * equation_of_time_approx)
        - 0.002697 * np.cos(3.0 * equation_of_time_approx)
        + 0.070257 * np.sin(equation_of_time_approx)
        + 0.000907 * np.sin(2.0 * equation_of_time_approx)
        + 0.001480 * np.sin(3.0 * equation_of_time_approx)
    )

    # equation of time, used to compensate for the earth's elliptical orbit
    # around the sun and its axial tilt when calculating solar time
    # eqt is the correction in hours
    equation_of_time = 2.0 * np.pi * day_of_year / 366.0
    equation_of_time = (
        0.0072 * np.cos(equation_of_time)
        - 0.0528 * np.cos(2.0 * equation_of_time)
        - 0.0012 * np.cos(3.0 * equation_of_time)
        - 0.1229 * np.sin(equation_of_time)
        - 0.1565 * np.sin(2.0 * equation_of_time)
        - 0.0041 * np.sin(3.0 * equation_of_time)
    )

    # calculate the solar zenith angle
    omega = np.radians(
        (360.0 / 24.0) * (hour_of_day + lon / 15.0 + equation_of_time - 12.0)
    )
    sunh = np.sin(solar_declination) * np.sin(np.radians(lat)) + np.cos(
        solar_declination
    ) * np.cos(np.radians(lat)) * np.cos(omega)

    solar_elevation = np.arcsin(np.clip(sunh, -1, 1))
    solar_zenith_angle = np.pi / 2.0 - solar_elevation

    # Solar azimuth added by yaswant
    azimuth = (
        np.sin(solar_declination) * np.cos(np.radians(lat))
        - np.cos(solar_declination) * np.sin(np.radians(lat)) * np.cos(omega)
    ) / np.cos(np.pi / 2.0 - solar_zenith_angle)

    solar_azimuth_angle = np.arccos(np.clip(azimuth, -1, 1))

    return np.degrees(solar_zenith_angle), np.degrees(solar_azimuth_angle)


def get_abi_proj(dataset: xr.Dataset) -> Proj:
    """
    Return a pyproj projection from the information contained within an ABI file
    """
    return Proj(
        proj="geos",
        h=dataset.goes_imager_projection.perspective_point_height,
        lon_0=dataset.goes_imager_projection.longitude_of_projection_origin,
        lat_0=dataset.goes_imager_projection.latitude_of_projection_origin,
        sweep=dataset.goes_imager_projection.sweep_angle_axis,
    )


def get_abi_x_y(
    lat: np.ndarray, lon: np.ndarray, dataset: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the x, y coordinates in the ABI projection for given latitudes and
        longitudes
    """
    p = get_abi_proj(dataset)
    x, y = p(lon, lat)
    return (
        x / dataset.goes_imager_projection.perspective_point_height,
        y / dataset.goes_imager_projection.perspective_point_height,
    )


def get_abi_lat_lon(
    dataset: xr.Dataset, dtype: type = float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns latitude and longitude for each location in an ABI dataset
    """
    p = get_abi_proj(dataset)
    xx, yy = np.meshgrid(
        (
            dataset.x.data * dataset.goes_imager_projection.perspective_point_height
        ).astype(dtype),
        (
            dataset.y.data * dataset.goes_imager_projection.perspective_point_height
        ).astype(dtype),
    )
    lons, lats = p(xx, yy, inverse=True)
    lons[lons >= 1e30] = np.nan
    lats[lats >= 1e30] = np.nan
    return lats, lons


def crop(
    data_dict: dict,
    patch_size: np.array,
    keys=["data", "coords", "sat_angle", "solar_angle"],
):
    """
    data_dict: dictionary containing the data, with keys 'data', 'coords', [optional] 'sat_angle', 'solar_angle'
    keys: list of keys to crop
    patch_size: tuple with the size of the patch to crop
    """
    # NOTE: might want to move this to transforms
    # TODO: add option to do (shifted) center crop instead of random crop for TC pretraining

    max_attempts = 20
    while True:
        # Get indexes to crop
        xmin = randint(
            0, data_dict["data"].shape[1] - patch_size[0]
        )  # Select random x index
        ymin = randint(
            0, data_dict["data"].shape[2] - patch_size[1]
        )  # Select random y index
        # Test if there are valid coordinates in the patch
        test_coords = data_dict["coords"][0][
            xmin : xmin + patch_size[0], ymin : ymin + patch_size[1]
        ]
        # check that no patch is fully off-disk
        if np.any(np.isfinite(test_coords)):
            break
        max_attempts -= 1
        if max_attempts == 0:
            logger.info(
                "Could not find valid coordinates in patch after 20 cropping attempts"
            )
            break

    # Crop the relevant variables
    for key in keys:
        data_dict[key] = data_dict[key][
            :, xmin : xmin + patch_size[0], ymin : ymin + patch_size[1]
        ]

    return data_dict


def expand_cloudsat_to3D(
    datatree: xr.DataTree,
    cloudsat_variables: list[str],
) -> np.ndarray:
    """
    Expand Cloudsat data to 3D array to match the dimensions of the goes satellite data.
    Do we want to stack multiple cloudsat variables together?
    Or add them as multiple items in the dictionary?
    """
    cloudsat_dict = {}

    for var in cloudsat_variables:
        cloudsat = datatree.cloudsat_aligned[var].values
        # Fill NaN values with the defined fill value
        if var not in CLOUDSAT_NAN_FILL_VALUES:
            raise ValueError(
                f"CloudSat variable {var} does not have a defined fill value in CLOUDSAT_NAN_FILL_VALUES"
            )
        cloudsat = np.where(np.isnan(cloudsat), CLOUDSAT_NAN_FILL_VALUES[var], cloudsat)
        y, x = datatree.geo_patch.cloudsat_overpass_mask.shape
        h = datatree.cloudsat_aligned.Nbin.size

        # Create array of NaNs with flattened horizontal axes
        out = np.full((y * x, h), np.nan)

        # Assign Cloudsat columns to spatial grid points using group dimension
        out[datatree.cloudsat_aligned["group"].values] = cloudsat

        # Reshape to patch size
        out = out.reshape((y, x, h))

        cloudsat_dict[var] = out
    return cloudsat_dict


def pad_track_with_nan(
    datatree: xr.DataTree, cloudsat_variables: list[str], target_length: int = 512
) -> np.ndarray:
    """
    Function to pad the CloudSat profiles with nans.
    """

    cloudsat_dict = {}

    for var in cloudsat_variables:
        if var in ["QR"]:
            cloudsat = datatree.cloudsat_aligned[var].sum("Nbands_Flxhr2B").values
        elif var in ["QR_sw"]:
            var = "QR"
            cloudsat = datatree.cloudsat_aligned[var][0].values
        elif var in ["QR_lw"]:
            var = "QR"
            cloudsat = datatree.cloudsat_aligned[var][1].values
        else:
            cloudsat = datatree.cloudsat_aligned[var].values

        if var not in CLOUDSAT_NAN_FILL_VALUES:
            raise ValueError(
                f"CloudSat variable {var} does not have a defined fill value in CLOUDSAT_NAN_FILL_VALUES"
            )
        cloudsat = np.where(np.isnan(cloudsat), CLOUDSAT_NAN_FILL_VALUES[var], cloudsat)

        # Extra dimensions (H: length, W: height of the track)
        H, W = cloudsat.shape

        # Add extra padding along the length dimension
        padded = np.full((target_length, W), np.nan, dtype=cloudsat.dtype)
        padded[:H, :] = cloudsat
        cloudsat_dict[var] = padded
    return cloudsat_dict


def safe_tensor_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Custom collate function that safely handles dictionaries containing numpy arrays
    and lists, converting them to tensors with proper error handling.

    This function addresses the "RuntimeError: Trying to resize storage that is not resizable"
    error by ensuring all tensors are created with resizable storage.
    """
    if not batch:
        return {}

    # Get the keys from the first item
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch]

        # Handle different data types
        if key in ["satellite", "band_names"]:
            # Keep as list for string/categorical data
            collated[key] = values
        elif key in ["time", "center_coords", "center_sat_angle", "center_solar_angle"]:
            # Handle scalar/1D arrays
            try:
                # Convert to numpy array first, then to tensor
                np_array = np.array(values)
                # Ensure contiguous memory layout
                np_array = np.ascontiguousarray(np_array)
                collated[key] = torch.tensor(np_array, dtype=torch.float32)
            except (ValueError, RuntimeError) as e:
                # If tensor conversion fails, keep as list
                print(f"Warning: Could not collate {key}, keeping as list: {e}")
                collated[key] = values
        elif key in ["data", "coords", "sat_angle", "solar_angle"]:
            # Handle multi-dimensional arrays
            try:
                # Check if all arrays have the same shape
                shapes = [v.shape for v in values]
                if len(set(shapes)) == 1:
                    # All same shape - can stack normally
                    np_array = np.stack(values, axis=0)
                    # Ensure contiguous memory layout
                    np_array = np.ascontiguousarray(np_array)
                    collated[key] = torch.tensor(np_array, dtype=torch.float32)
                else:
                    # Different shapes - pad to largest or keep as list
                    print(f"Warning: Inconsistent shapes for {key}: {shapes}")
                    collated[key] = values
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Could not collate {key}, keeping as list: {e}")
                collated[key] = values
        elif key == "cloudsat":
            # Handle nested dictionary for cloudsat data
            try:
                collated[key] = collate_cloudsat_dict(values)
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Could not collate cloudsat data, keeping as list: {e}")
                collated[key] = values
        elif key in ["wavelengths", "sensor_info"]:
            # Keep metadata as-is
            collated[key] = values[0]  # Assuming same for all samples
        elif key == "overpass_mask":
            # Handle boolean/mask data
            try:
                np_array = np.stack(values, axis=0)
                np_array = np.ascontiguousarray(np_array)
                collated[key] = torch.tensor(np_array, dtype=torch.bool)
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Could not collate {key}, keeping as list: {e}")
                collated[key] = values
        else:
            # Default: try to collate, fall back to list
            try:
                if isinstance(values[0], np.ndarray):
                    np_array = np.stack(values, axis=0)
                    np_array = np.ascontiguousarray(np_array)
                    collated[key] = torch.tensor(np_array, dtype=torch.float32)
                else:
                    collated[key] = values
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Could not collate {key}, keeping as list: {e}")
                collated[key] = values

    return collated


def collate_cloudsat_dict(
    cloudsat_list: list[dict[str, np.ndarray]]
) -> dict[str, torch.Tensor]:
    """
    Collate CloudSat data which is a dictionary of variables.
    """
    if not cloudsat_list:
        return {}

    # Get variables from first sample
    variables = cloudsat_list[0].keys()
    collated = {}

    for var in variables:
        var_data = [sample[var] for sample in cloudsat_list]

        try:
            # Check shapes
            shapes = [data.shape for data in var_data]
            if len(set(shapes)) == 1:
                # All same shape - can stack
                np_array = np.stack(var_data, axis=0)
                # Handle NaN values
                np_array = np.ascontiguousarray(np_array)
                collated[var] = torch.tensor(np_array, dtype=torch.float32)
            else:
                print(f"Warning: Inconsistent shapes for CloudSat {var}: {shapes}")
                collated[var] = var_data
        except (ValueError, RuntimeError) as e:
            print(f"Warning: Could not collate CloudSat {var}, keeping as list: {e}")
            collated[var] = var_data

    return collated


class CropDataset:
    def __init__(
        self,
        patch_size: tuple[int, int],
        center_crop: bool = False,
        radius: int = 0,  # Defined in pixels
    ):
        self.patch_size = patch_size
        self.center_crop = center_crop
        self.radius = radius

    def __call__(
        self,
        ds,
    ):
        if self.center_crop:
            # crop around the center of the image, randomly within radius if desired
            central_idxx = ds.sizes["x"] // 2
            central_idxy = ds.sizes["y"] // 2

            central_x = randint(central_idxx - self.radius, central_idxx + self.radius)
            central_y = randint(central_idxy - self.radius, central_idxy + self.radius)

        else:
            # crop randomly
            central_x = randint(
                self.patch_size[0] // 2, ds.sizes["x"] - self.patch_size[0] // 2
            )
            central_y = randint(
                self.patch_size[1] // 2, ds.sizes["y"] - self.patch_size[1] // 2
            )

        ds = ds.isel(
            x=slice(
                central_x - self.patch_size[0] // 2, central_x + self.patch_size[0] // 2
            ),
            y=slice(
                central_y - self.patch_size[1] // 2, central_y + self.patch_size[1] // 2
            ),
        )
        return ds
