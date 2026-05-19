from __future__ import annotations

import json
import os
from random import randint

import numpy as np
import scipy
import scipy.ndimage
import torch
from loguru import logger

from src.datamodules.constants import (
	GOES_WAVELENGTHS,
	HIMAWARI_WAVELENGTHS,
	MSG_WAVELENGTHS,
)
from src.datamodules.utils import standardize_patch_size
from src.senseiv2.encoding import SEnSeIv2EncodingFlex
from src.senseiv2.utils import (
	encode_position_angle,
	encode_position_spherical_angles,
	torch_circmean,
)


class SelectBandsTransform:
	"""
	Selects a subset of available bands from data dictionary
	"""

	def __init__(
		self,
		target_bands,
		keys=["data", "wavelengths", "sensor_info", "sensei_encoding"],
	):
		"""
		Args:
			target_bands (list): List of bands to select
			key (str): Key in dictionary to apply transformation
		"""
		self.target_bands = target_bands
		self.keys = keys

	def __call__(self, data_dict, **kwargs):
		source_bands = data_dict["band_names"]
		# Get indexes of bands to select
		indexes = [source_bands.index(band) for band in self.target_bands]

		for key in self.keys:
			# Extract data
			try:
				data = data_dict[key]
			except KeyError:
				continue
			if type(data) is list:
				data = [data[i] for i in indexes]
				assert len(data) == len(self.target_bands)
			elif type(data) is dict:
				data = {k: data[k] for k in self.target_bands}
			else:
				# Subselect bands
				data = data[indexes]
				assert data.shape[0] == len(self.target_bands)
			# Update dictionary
			data_dict[key] = data
		data_dict["band_names"] = self.target_bands
		return data_dict


class SelectWavelengthsTransform:
	"""
	Selects a subset of available bands from data dictionary
	"""

	def __init__(
		self,
		wavelengths,
		keys=["data", "wavelengths", "sensor_info", "sensei_encoding"],
	):
		"""
		Args:
			wavelengths (list): List of wavelengths to select closest matching band for
			key (str): Key in dictionary to apply transformation
		"""
		self.wavelengths = wavelengths
		self.keys = keys

	def __call__(self, data_dict, **kwargs):
		source_wavelengths = data_dict["wavelengths"]

		# match wavelengths to bands
		# find distance between self.wavelengths and source_wavelengths
		distances = np.abs(
			np.array(source_wavelengths)[:, None] - np.array(self.wavelengths)[None, :]
		)
		# for each, find the index of the closest wavelength
		indexes = np.argmin(distances, axis=0)

		closest_bands = [data_dict["band_names"][i] for i in indexes]
		wavelength_keys = [str(w) for w in self.wavelengths]

		for key in self.keys:
			# Extract data
			try:
				data = data_dict[key]
			except KeyError:
				continue
			if type(data) is list:
				data = [data[i] for i in indexes]
				assert len(data) == len(closest_bands)
			elif type(data) is dict:
				data = {w: data[k] for w, k in zip(wavelength_keys, closest_bands)}
				assert len(data) == len(closest_bands)
			else:
				# Subselect bands
				data = data[indexes]
				assert data.shape[0] == len(closest_bands)
			# Update dictionary
			data_dict[key] = data
		data_dict["band_names"] = wavelength_keys
		return data_dict


class SelectVariablesTransform:
	"""
	Selects a subset of available variables from a data dictionary.
	Stacks all data variables in "data" key.
	"""

	def __init__(self, vars, key="data"):
		"""
		Args:
			vars (list): List of variables to select
			key (str): Key in dictionary to apply transformation
		"""
		self.vars = vars
		self.key = key

	def __call__(self, data_dict, **kwargs):
		# Get data
		data = data_dict[self.key]
		# Expects data to be a dictionary
		assert isinstance(data, dict)
		# Expects variables to be a list
		assert isinstance(self.vars, list)
		# Select variables
		selection = [data[var] for var in self.vars]
		# Stack variables
		data = np.stack(selection, axis=0)
		# Update dictionary
		data_dict[self.key] = data
		return data_dict


class BandOrderTransform:
	"""
	Reorders bands in data dictionary.
	"""

	def __init__(self, target_order, key="data", band_info_key="wavelengths"):
		"""
		Args:
			target_order (list): Order of bands
			key (str): Key in dictionary to apply transformation
		"""
		self.target_order = target_order
		self.key = key
		self.band_info_key = band_info_key
		if self.band_info_key not in ["wavelengths", "band_names"]:
			raise ValueError(
				"band_info_key must be either 'wavelengths' or 'band_names'."
			)

	def __call__(self, data_dict, **kwargs):
		source_order = data_dict[self.band_info_key]
		assert len(source_order) == len(
			self.target_order
		), "Length of source and target wavelengths must match."
		# if type(source_order) != np.ndarray:
		#     source_order = np.array(source_order)
		# if type(self.target_order) != np.ndarray:
		#     self.target_order = np.array(self.target_order)
		# Get indexes of bands to select
		# TODO: Check if source_order is a list or numpy array
		indexes = [source_order.index(wvl) for wvl in self.target_order]
		# Extract data
		data = data_dict[self.key]
		# Subselect bands
		data = data[indexes]
		# Update dictionary
		data_dict[self.key] = data
		# Update wavelengths
		if "wavelengths" in data_dict.keys():
			source_wavelengths = data_dict["wavelengths"]
			target_wavelengths = [source_wavelengths[i] for i in indexes]
			data_dict["wavelengths"] = target_wavelengths
		# Update band names
		if (
			"band_names" in data_dict.keys()
		):  # Band names were previously not included for MSG. This has now been changed.
			source_band_names = data_dict["band_names"]
			target_band_names = [source_band_names[i] for i in indexes]
			data_dict["band_names"] = target_band_names
		return data_dict


class NanMaskTransform:
	# TODO not yet tested
	"""
	Returns mask for NaN values in data dictionary
	"""

	def __init__(self, key="data"):
		self.key = key

	def __call__(self, data_dict, **kwargs):
		data = data_dict[self.key]
		# Check if any band contains NaN values
		mask = np.isnan(data).any(axis=0)
		mask = mask.astype(int)
		# Update dictionary
		data_dict["nan_mask"] = mask
		return data_dict


class NanDictTransform:
	# TODO not yet tested
	"""
	Removes NaN values from data dictionary.
	Can also be used to replace NaN values of coordinates to remove off limb data.
	"""

	def __init__(self, key="data", fill_value=0):
		self.key = key
		self.fill_value = fill_value

	def __call__(self, data_dict, **kwargs):
		data = data_dict[self.key]
		# Replace NaN values
		data = np.nan_to_num(data, nan=self.fill_value)
		# Update dictionary
		data_dict[self.key] = data
		return data_dict


class NanDictPatchMeanTransform:
	"""
	Replaces NaN values in data dictionary with channel mean of that patch.
	"""

	def __init__(self, key="data"):
		self.key = key

	def __call__(self, data_dict, **kwargs):
		data = data_dict[self.key]
		# Replace NaN values
		for i in range(data.shape[0]):
			data[i] = np.nan_to_num(data[i], nan=np.nanmean(data[i]))
		# Update dictionary
		data_dict[self.key] = data
		return data_dict


class ReplaceValueTransform:
	"""
	Replaces a value in data dictionary with another value.
	"""

	def __init__(self, value, new_value, key="data"):
		self.value = value
		self.new_value = new_value
		self.key = key

	def __call__(self, data_dict, **kwargs):
		data = data_dict[self.key]
		# Replace value
		data[data == self.value] = self.new_value
		# Update dictionary
		data_dict[self.key] = data
		return data_dict


class CoordNormTransform:
	"""
	Normalize latitude and longitude coordinates
	"""

	def __init__(self, key="coords", lat_range=[-90, 90], lon_range=[-180, 180]):
		self.key = key
		self.lat_range = lat_range
		self.lon_range = lon_range

	def __call__(self, data_dict, **kwargs):
		lats, lons = data_dict["coords"]

		# Normalize latitude and longitude to range [-1, 1]
		# shift centred around 0
		lats = lats - (self.lat_range[0] + self.lat_range[1]) / 2
		lons = lons - (self.lon_range[0] + self.lon_range[1]) / 2
		# scale to [-1, 1]
		lats = lats / ((self.lat_range[1] - self.lat_range[0]) / 2)
		lons = lons / ((self.lon_range[1] - self.lon_range[0]) / 2)

		# Update dictionary
		data_dict["coords"] = np.stack([lats, lons], axis=0)
		return data_dict


class MeanStdNormaliseTransform:
	# TODO add a check to make sure that we have all the means and stds for radiances
	"""
	Normalises data to have zero mean and unit variance.
	"""

	def __init__(self, band_info_path, key="data", band_info_key="wavelengths"):
		"""
		Args:
			band_infi (dict): Dictionary containing mean and std for each band
			key (str): Key in dictionary to apply transformation
		"""
		self.band_info_path = band_info_path
		self.key = key
		self.band_info_key = band_info_key

	def __call__(self, data_dict, **kwargs):
		# get data to be normalised
		data = data_dict[self.key]
		with open(os.path.join(os.path.expanduser("~"), self.band_info_path)) as f:
			self.band_info = json.load(f)
		# get the mean and std for each band - type conversion needed as json keys and values are strings
		means = np.array(
			[
				float(self.band_info[str(key)]["mean"])
				for key in data_dict[self.band_info_key]
			]
		)
		stds = np.array(
			[
				float(self.band_info[str(key)]["std"])
				for key in data_dict[self.band_info_key]
			]
		)
		# normalise each band using mean and std
		data = (data - means[:, None, None]) / stds[:, None, None]
		# update dictionary
		data_dict[self.key] = data
		return data_dict


class StackDictTransformSimple:
	"""
	Stack data dictionary into a single array
	"""

	def __init__(
		self,
		keys=["data", "coords", "sat_angle", "solar_angle", "time"],
		stack_key="data",
		axis=0,
	):
		self.keys = keys
		self.stack_key = stack_key
		self.axis = axis

	def __call__(self, data_dict, **kwargs):
		# Select data
		data = []
		for key in self.keys:
			values = data_dict[key]
			if len(values.shape) == 2:
				# if the variable has no channel dimension, add a new axis
				values = np.expand_dims(values, axis=self.axis)
			data.append(values)
		# Stack data
		data = np.concatenate(data, axis=self.axis)
		# Update dictionary
		data_dict[self.stack_key] = data
		# Return numpy array
		return data


class StackDictTransform:
	"""
	Stack data dictionary into a single array
	"""

	def __init__(
		self,
		keys=["data", "coords", "sat_angle", "solar_angle", "time"],
		stack_key="data",
		only_fractional_time=True,
		norm_angles=False,
		axis=0,
	):
		self.keys = self.__rename__(keys)
		self.axis = axis
		self.stack_key = stack_key
		self.only_fractional_time = only_fractional_time
		self.norm_angles = norm_angles  # whether to normalize the angles between [0, 1] or keep as [0, 2*pi]
		self.convert2radians = ConvertToRadiansTransform(norm_angles=self.norm_angles)
		self.convert2d = TimeTo2DTransform()

	def __rename__(self, keys):
		renamed_keys = []
		for key in keys:
			# NOTE: Angles and time are saved as new keys to avoid overwriting original values
			if key in ["coords", "sat_angle", "solar_angle"]:
				renamed_keys.append(f"{key}_rad")
			elif key == "time":
				renamed_keys.append(f"{key}_2d")
			else:
				renamed_keys.append(key)
		return renamed_keys

	def __call__(self, data_dict):
		# Convert angles to radians
		data_dict = self.convert2radians(data_dict)
		# Convert 1D time arrays to 2D arrays
		data_dict = self.convert2d(data_dict)
		# Select data
		data = []

		# NOTE: Needed if some keys are not shaped ([...], 256, 256)
		data_dict = standardize_patch_size(data_dict)

		for key in self.keys:
			values = data_dict[key]
			if len(values.shape) == 2:
				# if the variable has no channel dimension, add a new axis
				values = np.expand_dims(values, axis=self.axis)
			if "time" in key and self.only_fractional_time:
				# if the variable is time, only keep the fraction of year/day
				# if only_fractional_time is False, all values are kept
				values = values[-2:, :, :]
			data.append(values)
		# Stack data
		data = np.concatenate(data, axis=self.axis).astype(np.float32)
		# Update dictionary
		data_dict[self.stack_key] = data
		# Return numpy array
		return data_dict


class ConvertToRadiansTransform:
	"""
	Convert angles in degrees to radians.
	"""

	def __init__(
		self,
		keys=["coords", "sat_angle", "solar_angle"],
		norm_angles: bool = False,
	):
		self.keys = keys
		self.norm_angles = norm_angles
		self.ranges = {
			"coords": {
				"zenith": {  # scaling for latitude
					"min": -90,
					"max": 90,
				},
				"azimuth": {  # scaling for longitude
					"min": -180,
					"max": 180,
				},
			},
			"sat_angle": {
				"zenith": {
					"min": 0,
					"max": 180,
				},
				"azimuth": {
					"min": 0,
					"max": 360,
				},
			},
			"solar_angle": {
				"zenith": {
					"min": 0,
					"max": 180,
				},
				"azimuth": {
					"min": 0,
					"max": 360,
				},
			},
		}

	def convert_angle(self, data, min, max):
		"""
		Convert angles in degrees to radians and scale to [0, 2*pi].
		"""
		# convert to radians and scale to [0, 2*pi]
		val_radians = 2 * np.pi * (data - min) / (max - min)
		return val_radians

	def convert_half_angle(self, data, min, max):
		"""
		Convert angles in degrees to radians and scale to [0, pi].
		"""
		# convert to radians and scale to [0, pi]
		val_radians = np.pi * (data - min) / (max - min)
		return val_radians

	def __call__(self, data_dict):
		# Convert angles to radians
		for key in self.keys:
			data = data_dict[key]
			data_0 = self.convert_half_angle(
				data[0],
				self.ranges[key]["zenith"]["min"],
				self.ranges[key]["zenith"]["max"],
			)
			data_1 = self.convert_angle(
				data[1],
				self.ranges[key]["azimuth"]["min"],
				self.ranges[key]["azimuth"]["max"],
			)
			if self.norm_angles:  # normalize between [0, 1] if self.norm
				data_0 = data_0 / (2 * np.pi)
				data_1 = data_1 / (2 * np.pi)
			# Update dictionary
			# NOTE: saving with a new key to avoid overwriting original angles
			data_dict[f"{key}_rad"] = np.stack([data_0, data_1], axis=0)
		return data_dict


class TimeTo2DTransform:
	"""
	Copy 1D arrays to 2D arrays of shape (1, H, W).
	"""

	def __init__(self, keys=["time"], height=256, width=256):
		self.keys = keys
		self.height = height
		self.width = width

	def __call__(self, data_dict):
		for key in self.keys:
			length = len(data_dict[key])
			data_2d = np.zeros(
				(length, self.height, self.width), dtype=data_dict[key].dtype
			)
			for i in range(length):
				data_2d[i, :, :] = np.tile(
					data_dict[key][i], (1, self.height, self.width)
				)
			# NOTE: saving with a new key to avoid overwriting original times
			data_dict[f"{key}_2d"] = data_2d
		return data_dict


class ToTensorTransform:
	"""
	Convert numpy array to PyTorch tensor
	"""

	def __init__(self, dtype=torch.float32):
		self.dtype = dtype

	def __call__(self, data, **kwargs):
		# Convert to tensor
		tensor = torch.as_tensor(data, dtype=self.dtype)
		return tensor


class CropHeightTransform:
	"""
	Crop height levels, e.g. for Cloudsat data.
	"""

	def __init__(self, bottom_cutoff: int, top_cutoff: int, key="cloudsat"):
		self.bottom_cutoff = bottom_cutoff
		self.top_cutoff = top_cutoff
		self.key = key

	def __call__(self, data_dict, **kwargs):
		# Crop height levels
		for var_key in data_dict[self.key]:
			data_dict[self.key][var_key] = data_dict[self.key][var_key][
				...,
				self.top_cutoff : -self.bottom_cutoff,  # CloudSat profiles go top to bottom
			]
		return data_dict


class RandomCropDictTransform:
	def __init__(
		self,
		patch_size: tuple[int, int],
		center_crop: bool = False,
		radius: int = 0,  # Defined in pixels
		keys: list[str] = ["data", "coords", "sat_angle", "solar_angle"],
		max_attempts: int = 10,
	):
		self.patch_size = patch_size
		self.center_crop = center_crop
		self.radius = radius
		self.keys = keys
		self.max_attempts = max_attempts

	def __call__(
		self,
		data_dict,
	):
		# first randomly select cropping pixels
		# NOTE: this function assumes that all values in data_dict are C x H x W and have the same H x W

		if self.center_crop:
			# crop around the center of the image, randomly within radius if desired
			central_idxx = data_dict[self.keys[0]].shape[1] // 2
			central_idxy = data_dict[self.keys[0]].shape[2] // 2

			central_x = randint(central_idxx - self.radius, central_idxx + self.radius)
			central_y = randint(central_idxy - self.radius, central_idxy + self.radius)

		else:
			while self.max_attempts > 0:
				# randomly select cropping pixels
				xmin = randint(0, data_dict[self.keys[0]].shape[1] - self.patch_size[0])
				ymin = randint(0, data_dict[self.keys[0]].shape[2] - self.patch_size[1])
				# check that patch was not cropped off disk
				test_coords = data_dict["coords"][
					:,
					xmin : xmin + self.patch_size[0],
					ymin : ymin + self.patch_size[1],
				]
				if np.isnan(test_coords).all() or np.isinf(test_coords).all():
					max_attempts -= 1
					if max_attempts == 0:
						logger.warning(
							f"Could not find valid patch after {self.max_attempts} cropping attempts ... "
						)
					else:
						continue
				else:
					break  # patch is valid, break loop

		# then crop arrays for all keys using the same cropping pixels
		for key in self.keys:
			assert (
				data_dict[key].shape[1] >= self.patch_size[0]
			), f"Invalid shape to crop: data shape 1 is {data_dict[key].shape[1]}, patch size 0 is {self.patch_size[0]}"
			assert (
				data_dict[key].shape[2] >= self.patch_size[1]
			), f"Invalid shape to crop: data shape 2 is {data_dict[key].shape[2]}, patch size 1 is {self.patch_size[1]}"

			if not self.center_crop:
				data_dict[key] = data_dict[key][
					:,
					xmin : xmin + self.patch_size[0],
					ymin : ymin + self.patch_size[1],
				]
			else:
				data_dict[key] = data_dict[key][
					:,
					central_x
					- self.patch_size[0] // 2 : central_x
					+ self.patch_size[0] // 2,
					central_y
					- self.patch_size[1] // 2 : central_y
					+ self.patch_size[1] // 2,
				]
		return data_dict


class RandomCropTransform:
	# NOTE: This outputs an array, above RandomCropDictTransform outputs a dictionary
	# Might want to decide on one

	def __init__(
		self,
		patch_size,
		center_crop=False,
		radius=0,  # Defined in pixels
	):
		self.patch_size = patch_size
		self.center_crop = center_crop
		self.radius = radius

	def __call__(self, arr):
		assert arr.shape[1] >= self.patch_size[0], "Invalid shape to crop"
		assert arr.shape[2] >= self.patch_size[1], "Invalid shape to crop"
		if not self.center_crop:
			xmin = randint(0, arr.shape[1] - self.patch_size[0])
			ymin = randint(0, arr.shape[2] - self.patch_size[1])
			cropped_arr = arr[
				:, xmin : xmin + self.patch_size[0], ymin : ymin + self.patch_size[1]
			]
		else:
			central_idxx, central_idxy = arr.shape[1] // 2, arr.shape[2] // 2
			central_x = randint(central_idxx - self.radius, central_idxx + self.radius)
			central_y = randint(central_idxy - self.radius, central_idxy + self.radius)
			cropped_arr = arr[
				:,
				central_x
				- self.patch_size[0] // 2 : central_x
				+ self.patch_size[0] // 2,
				central_y
				- self.patch_size[1] // 2 : central_y
				+ self.patch_size[1] // 2,
			]

		return cropped_arr


class CloudSatLinearNormaliseTransform:
	"""
	Normalizes variables to [-1, 1].
	"""

	def __init__(
		self,
		var: str,
		min: float,
		max: float,
		key: str = "cloudsat",
	):
		"""
		Args:
			var (list): The radar product to select
			key (str): Key in dictionary to apply transformation
		"""
		self.var = var
		self.key = key
		self.min = min
		self.max = max

	def __call__(self, data_dict, **kwargs):
		# Get data
		data = data_dict[self.key][self.var]
		# clip extreme values
		data = np.clip(data, self.min, self.max)
		# apply normalization to valid data
		normalized_data = np.empty_like(data, dtype=float)
		normalized_data = (2 * (data - self.min) / (self.max - self.min)) - 1
		data_dict[self.key][self.var] = normalized_data
		return data_dict


class CloudSatLinearUnormaliseTransform:
	def __init__(self, min, max):
		self.min = min
		self.max = max

	def __call__(self, array):
		# Unnormalize the array
		unnormalized_array = ((array + 1) / 2) * (self.max - self.min) + self.min
		return unnormalized_array


class CloudSatQRLinearNormaliseTransform:
	"""
	Normalizes variables to [-1, 1].
	"""

	def __init__(
		self,
		var: str,
		min: float,
		max: float,
		key: str = "cloudsat",
	):
		"""
		Args:
			var (list): The radar product to select
			key (str): Key in dictionary to apply transformation
		"""
		self.var = var
		self.key = key
		self.min = min
		self.max = max

	def __call__(self, data_dict, **kwargs):
		# Get data
		data = data_dict[self.key][self.var]
		# clip extreme values
		data = np.clip(data, self.min, self.max)
		# apply normalization to valid data
		normalized_data = np.empty_like(data, dtype=float)
		normalized_data = (2 * (data - self.min) / (self.max - self.min)) - 1
		data_dict[self.key][self.var] = normalized_data
		return data_dict


class CloudSatLogNormaliseTransform:
	"""
	Log normalizes CloudSat variables and scales to [-1, 1].
	"""

	def __init__(
		self,
		var: str,
		key: str = "cloudsat",
		min=1e-4,
		max=100,
	):
		"""
		Args:
			var (list): The radar product to select
			key (str): Key in dictionary to apply transformation
		"""
		self.var = var
		self.key = key
		self.min = min
		self.max = max

	def __call__(self, data_dict, **kwargs):
		# Get data
		data = data_dict[self.key][self.var]
		# clip extreme values and take log
		data = np.log10(np.clip(data, self.min, self.max))
		# apply normalization to valid data
		normalized_data = np.empty_like(data)
		normalized_data = (
			2
			* ((data - np.log10(self.min)) / (np.log10(self.max) - np.log10(self.min)))
			- 1
		).astype(data.dtype)
		data_dict[self.key][self.var] = normalized_data
		return data_dict


class CloudSatLogUnnormaliseTransform:
	def __init__(self, min, max):
		self.min = min
		self.max = max

	def __call__(self, array):
		# Unnormalize the array
		unnormalized_array = ((array + 1) / 2) * (
			np.log10(self.max) - np.log10(self.min)
		) + np.log10(self.min)
		# use torch.pow if array is a tensor
		if isinstance(array, torch.Tensor):
			unnormalized_array = torch.pow(
				10, unnormalized_array
			)  # Convert back from log scale
		else:
			unnormalized_array = np.power(
				10, unnormalized_array
			)  # Convert back from log scale
		return unnormalized_array


class CloudsatFillMeasurementGapsTransform:
	"""
	Fill NaN rows in Cloudsat data by copying the nearest non-NaN row above or below.
    
	nearest: performs nearest neighbour interpolation to fill NaN rows.
	linear: performs linear interpolation of the surrounding values to fill NaN rows.
    
	call after RadarReflectivityNormaliseTransform.
	"""

	def __init__(
		self,
		var: str = "Radar_Reflectivity_Fwd",
		key: str = "data",
		max_gap: int = 15,
		interpolation: str = "linear",
	):
		"""
		Args:
			key (str): Key in dictionary to apply transformation
		"""
		self.var = var
		self.key = key
		self.max_gap = max_gap
		self.interpolation = interpolation

	def _linear_interp(self, data):
		"""
		replaces nan values with linear interpolation of surrounding values
		"""
		bad_indexes = np.isnan(data)
		good_indexes = np.logical_not(bad_indexes)
		good_data = data[good_indexes]
		interpolated = np.interp(
			bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data
		)
		data[bad_indexes] = interpolated
		return data

	def _remove_too_large_gaps(self, nan_rows):
		"""
		finds continuous gaps in nan_rows that are larger than self.max_gap and removes them
		"""
		i = 0
		gaps = []
		while i < len(nan_rows):
			if (
				i + self.max_gap < len(nan_rows)
				and nan_rows[i + self.max_gap] - nan_rows[i] == self.max_gap
			):
				# we know the gap is at least self.max_gap rows long
				# find all indices that are part of this continuous gap
				gap = []
				for j in range(i, len(nan_rows)):
					if i + j < len(nan_rows) and nan_rows[j + 1] - nan_rows[j] == 1:
						gap.append(nan_rows[j])
					else:
						# nan_rows[j] is the last index that is part of the continuous gap
						gap.append(nan_rows[j])
						break
				gaps += gap
				i = j + 1
			else:
				i += 1

		# remove indices in gaps from nan_rows
		nan_rows = np.array([i for i in nan_rows if i not in gaps])
		return nan_rows

	def _fill_nan_rows(self, data):
		"""
		Fills NaN rows in data using either nearest neighbour or linear interpolation.
		"""
		n_rows, _ = data.shape
		if self.interpolation == "nearest":
			"""
			Fill NaN rows by copying the nearest non-NaN row above or below.
			"""
			max_distance = self.max_gap // 2
			for i in range(n_rows):
				if np.isnan(data[i]).all():  # Check if the whole row is NaN
					# Find closest row with real values within max_distance
					for d in range(1, max_distance + 1):
						# Check both directions: previous (i-d) and next (i+d)
						if i - d >= 0 and not np.isnan(data[i - d]).all():
							data[i] = data[i - d]
							break
						if i + d < n_rows and not np.isnan(data[i + d]).all():
							data[i] = data[i + d]
							break
		elif self.interpolation == "linear":
			"""
			Fill NaN rows by linear interpolation of the surrounding columns.
			"""
			# set nans in data to -1 as this corresponds to 0 reflectivity
			data[np.isnan(data)] = -1

			# get rows in data that are all -1
			nan_rows = np.where((data == -1).all(axis=1))[0]

			# remove continuous gaps in nan_rows that are larger than self.max_gap
			nan_rows = self._remove_too_large_gaps(nan_rows)

			# set items in rows to nan
			data[nan_rows] = np.nan

			# apply linear interpolation to each row
			data = np.apply_along_axis(self._linear_interp, 0, data)

		return data

	def __call__(self, data_dict, **kwargs):
		data = data_dict[self.key][self.var]
		# replace -999 in data with nan for fill_nan_rows
		data[data == -999] = np.nan
		data = self._fill_nan_rows(data)
		# there shouldn't be any nans left... replace nan with -1 for consistency
		data[np.isnan(data)] = -1
		data_dict[self.key][self.var] = data
		return data_dict


class UpDownSampleTransform:

	"""
	Increase (Upsample) or decrease (Downsample) the resolution of the data.
    
	NOTE: if we decrease resolution, we need to call this transform before cropping
	"""

	def __init__(
		self,
		source_reso: float,
		target_reso: float,
		keys: list[str] = ["data", "coords", "sat_angle", "solar_angle"],
	):
		# calculate scale factor from source and target resolution
		scale_factor = source_reso / target_reso
		self.zoom = (1.0, scale_factor, scale_factor)  # (C, H, W) for 2D data
		self.keys = keys

	def __call__(self, data_dict):
		for key in self.keys:
			data_dict[key] = scipy.ndimage.zoom(
				data_dict[key],
				zoom=self.zoom,
				order=1,  # bilinear interpolation
			)

		# TODO: Change resolution in wavelengths dict for sensei
		# for key in self.data_dict["wavelengths"]:
		# orig_res = data_dict["wavelengths"]["resolution"]
		# data_dict["wavelengths"]["resampled_resolution"] = orig_res*scale_factor

		return data_dict


class MinMaxNormaliseTransform:
	"""
	Normalises data to a range of [-1, 1] using min-max scaling.
	"""

	def __init__(self, bt_min=180, bt_max=350, nr_min=0, nr_max=100):
		self.bt_min = bt_min
		self.bt_max = bt_max
		self.nr_min = nr_min
		self.nr_max = nr_max

	def __call__(self, data_dict, **kwargs):
		for i, key in enumerate(data_dict["band_names"]):
			sensor_type = data_dict["sensor_info"][key]["band_type"]
			if sensor_type == "TOA Normalised Brightness Temperature":
				data_dict["data"][i] = np.clip(
					data_dict["data"][i], self.bt_min, self.bt_max
				)
				# Apply min-max scaling to [-1, 1]
				data_dict["data"][i] = (
					(data_dict["data"][i] - self.bt_min)
					/ (self.bt_max - self.bt_min)
					* 2
				) - 1
			if sensor_type == "TOA Reflectance":
				data_dict["data"][i] = np.clip(
					data_dict["data"][i], self.nr_min, self.nr_max
				)
				# Apply min-max scaling to [-1, 1]
				data_dict["data"][i] = (
					(data_dict["data"][i] - self.nr_min)
					/ (self.nr_max - self.nr_min)
					* 2
				) - 1
		return data_dict


class SenseiEncodingTransform:
	"""
	Adds a SEnSeI encoding token to the batch to be input to sensei embedding
	"""

	def __init__(self, satellite: str, embed_sizes_dict: dict[int]):
		self.embed_sizes_dict = embed_sizes_dict
		static_encoding_keys = [
			"min_wavelength",
			"max_wavelength",
			"center_wavelength",
			"reso_og",
		]
		self.static_encoding_sizes = {
			k: v for k, v in self.embed_sizes_dict.items() if k in static_encoding_keys
		}
		self.dynamic_encoding_sizes = {
			k: v
			for k, v in self.embed_sizes_dict.items()
			if k not in static_encoding_keys
		}
		self.satellite = satellite
		self._get_static_encoding()

	def __call__(self, data_dict):
		if self.dynamic_encoding_sizes:
			try:
				data_dict["sensei_encoding"] = self._get_combined_encodings(data_dict)
			except RuntimeError:
				data_dict["sensei_encoding"] = self.static_encoding
		else:
			data_dict["sensei_encoding"] = self.static_encoding
		return data_dict

	def _get_static_encoding(self):
		match self.satellite.lower():
			case "goes":
				sensor_constants = GOES_WAVELENGTHS
			case "himawari":
				sensor_constants = HIMAWARI_WAVELENGTHS
			case "msg":
				sensor_constants = MSG_WAVELENGTHS
			case _:
				raise ValueError(
					f'SenseiEncodingTransform error: satellite parameter must be one of ["goes", "himawari", "msg"], received satellite={self.satellite}'
				)
		self.static_encoding = SEnSeIv2EncodingFlex(
			emb_sizes=self.static_encoding_sizes
		)(
			{
				k: {
					vk: torch.asarray([vv]) if not isinstance(vv, str) else [vv]
					for vk, vv in v.items()
				}
				for k, v in sensor_constants.items()
			}
		).squeeze()
		# if len(self.static_encoding) > 2:
		#     self.static_encoding = self.static_encoding.squeeze()

	def _get_dynamic_encoding(self, data_dict):
		encodings = []
		if coord_dim := self.dynamic_encoding_sizes.get("coords", False):
			encodings.append(
				encode_position_spherical_angles(
					torch.nanmean(torch.asarray(data_dict["coords"][0]), dim=(-2, -1)),
					torch_circmean(
						torch.asarray(data_dict["coords"][1]),
						low=0,
						high=360,
						dim=(-2, -1),
					),
					encode_dim=coord_dim,
					zenith_min=-90,
					zenith_max=90,
					azimuth_min=-180,
					azimuth_max=180,
				)
			)

		if sat_angle_dim := self.dynamic_encoding_sizes.get("sat_angle", False):
			encodings.append(
				encode_position_spherical_angles(
					torch.nanmean(
						torch.asarray(data_dict["sat_angle"][0]), dim=(-2, -1)
					),
					torch_circmean(
						torch.asarray(data_dict["sat_angle"][1]),
						low=0,
						high=360,
						dim=(-2, -1),
					),
					encode_dim=sat_angle_dim,
					zenith_min=0,
					zenith_max=180,
					azimuth_min=0,
					azimuth_max=360,
				)
			)

		if solar_angle_dim := self.dynamic_encoding_sizes.get("solar_angle", False):
			encodings.append(
				encode_position_spherical_angles(
					torch.nanmean(
						torch.asarray(data_dict["solar_angle"][0]), dim=(-2, -1)
					),
					torch_circmean(
						torch.asarray(data_dict["solar_angle"][1]),
						low=0,
						high=360,
						dim=(-2, -1),
					),
					encode_dim=solar_angle_dim,
					zenith_min=0,
					zenith_max=180,
					azimuth_min=0,
					azimuth_max=360,
				)
			)

		if time_dim := self.dynamic_encoding_sizes.get("time", False):
			encodings.append(
				encode_position_angle(
					torch.asarray(data_dict["time"][-2]),
					encode_dim=time_dim // 2,
					min=0,
					max=1,
				)
			)
			encodings.append(
				encode_position_angle(
					torch.asarray(data_dict["time"][-1]),
					encode_dim=time_dim // 2,
					min=0,
					max=1,
				)
			)

		return torch.cat(encodings, dim=-1)

	def _get_combined_encodings(self, data_dict):
		dynamic_encoding = self._get_dynamic_encoding(data_dict)
		channels = self.static_encoding.shape[0]

		return torch.cat(
			[dynamic_encoding.repeat(channels, 1), self.static_encoding], dim=-1
		)

