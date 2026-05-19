from __future__ import annotations

import autoroot  # required to import from src
from torchvision.transforms import Compose

from src.transforms.transforms import (
    CloudSatLinearNormaliseTransform,
    CloudSatLogNormaliseTransform,
    CropHeightTransform,
    MinMaxNormaliseTransform,
    NanDictTransform,
    SelectBandsTransform,
    SelectWavelengthsTransform,
    SenseiEncodingTransform,
    StackDictTransform,
)


class CloudsatGEOTransform:

    """Transform pipeline for cloudsat aligned with geo sats."""

    def __init__(
        self,
        target_bands: None
        | (
            list
        ) = None,  # NOTE: this expect a list of strings for bandnames of geo sats
        target_wavelengths: None
        | (list) = None,  # NOTE: This expects a list of floats with wavelengths in nm
        cloudsat_variables=["Radar_Reflectivity"],
        satellite: str | None = None,
        embed_sizes_dict: dict | None = None,
        stack_keys: list[str] = None,  # if not None, stack the keys in the data_dict
    ):
        transform_list = []

        # Geostationary Satellite Transforms
        if embed_sizes_dict is not None:
            transform_list.append(
                SenseiEncodingTransform(
                    satellite=satellite, embed_sizes_dict=embed_sizes_dict
                )
            )
        if target_bands is not None:
            transform_list.append(SelectBandsTransform(target_bands=target_bands))
        elif target_wavelengths is not None:
            transform_list.append(
                SelectWavelengthsTransform(wavelengths=target_wavelengths)
            )

        transform_list += [MinMaxNormaliseTransform()]
        transform_list += [NanDictTransform(key="data")]

        # CloudSat Transforms
        if "Radar_Reflectivity" in cloudsat_variables:
            transform_list += [
                CloudSatLinearNormaliseTransform(
                    var="Radar_Reflectivity", min=-30, max=20
                )
            ]
        # NOTE: Implement other CloudSat normalizations
        # for var in cloudsat_variables:
        #     transforms_list += [CloudSatVariableNormaliseTransform(var)]

        if "IWC" in cloudsat_variables:
            transform_list += [
                CloudSatLogNormaliseTransform(var="IWC", min=1e-5, max=10)
                # CloudSatLinearNormaliseTransform(var="IWC", min=1e-5, max=10)
            ]

        if "RO_liq_water_content" in cloudsat_variables:
            transform_list += [
                CloudSatLogNormaliseTransform(
                    var="RO_liq_water_content", min=1e-1, max=100000
                )
            ]
        if "re" in cloudsat_variables:
            transform_list += [
                CloudSatLinearNormaliseTransform(var="re", min=0, max=160)
            ]

        if "QR" in cloudsat_variables or "QR_lw" in cloudsat_variables:
            transform_list.append(
                CloudSatLinearNormaliseTransform(var="QR", min=-50, max=50)
            )

        # If you want to stack keys to pass to the model
        if stack_keys is not None:
            transform_list += [
                StackDictTransform(
                    keys=stack_keys,
                    stack_key="data",
                    axis=0,
                    norm_angles=True,  # whether to normalize the angles between [0, 1]
                )
            ]

        transform_list += [
            CropHeightTransform(bottom_cutoff=20, top_cutoff=25),
        ]

        self.transform = Compose(transform_list)

    def __call__(self, sample):
        s = self.transform(sample)
        return s

