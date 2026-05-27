from __future__ import annotations

from typing import Callable

import autoroot  # required to load from src
from lightning.pytorch import LightningDataModule

from src.datamodules.cloudsat_goes_dataloader import CloudsatGOESDataModule
from src.datamodules.cloudsat_himawari_dataloader import CloudsatHIMAWARIDataModule
from src.datamodules.cloudsat_msg_dataloader import CloudsatMSGDataModule
from src.datamodules.constants import SPLITS_DICT
from src.datamodules.multi_dataloader import OneSatellitePerBatchDataLoader


class CloudSatMultiDataModule(LightningDataModule):
    """ LightningDataModule combining CloudSat-paired MSG, GOES, and Himawari data modules. """

    def __init__(
        self,
        data_dir_dict: dict[str, str],
        splits_dict=SPLITS_DICT,
        satellites: list[str] = ["msg_cloudsat", "goes_cloudsat", "himawari_cloudsat"],
        transforms_dict: dict[str, Callable] | None = None,
        ext: str = "nc",
        batch_size: int = 4,
        num_workers: int = 1,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        load_overpass_mask: bool = True,  # defaults to False for pre-training
        load_zenith: bool = True,
        load_solar: bool = True,
        cloudsat_variables: list[str] = ["Radar_Reflectivity"],
        file_number: int = None,
        filter_clear_sky: dict | None = None,
        weight_satellites: bool = False
    ):
        """ Initialize CloudSatMultiDataModule.

            Parameters
            ----------
            data_dir_dict : dict[str, str]. Mapping from satellite name to its data directory path.
            splits_dict : dict. Dictionary specifying train/test/val split criteria (optional).
            satellites : list[str]. List of satellite keys; must match data_dir_dict keys (optional).
            transforms_dict : dict[str, Callable] | None. Per-satellite transform callables (optional).
            ext : str. File extension to search for in data directories (optional).
            batch_size : int. Number of samples per batch (optional).
            num_workers : int. Number of DataLoader worker processes (optional).
            pin_memory : bool. If True, pins tensors to memory for faster GPU transfer (optional).
            prefetch_factor : int. Number of batches to prefetch per worker (optional).
            load_overpass_mask : bool. If True, loads the CloudSat overpass mask (optional).
            load_zenith : bool. If True, loads zenith angle data (optional).
            load_solar : bool. If True, loads solar angle data (optional).
            cloudsat_variables : list[str]. CloudSat variable names to load (optional).
            file_number : int. If set, randomly subsamples this many files from the full list (optional).
            filter_clear_sky : dict | None. Parameters for clear-sky scene filtering; None disables filtering (optional).
            weight_satellites : bool. If True, samples satellites proportionally to dataset size (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.data_dir_dict = data_dir_dict
        self.satellites = satellites
        self.weight_satellites = weight_satellites

        assert set(satellites) == set(
            data_dir_dict.keys()
        ), "Mismatch between satellites list and filenames_dict keys"
        if transforms_dict:
            assert set(satellites) == set(
                transforms_dict.keys()
            ), "Mismatch between satellites list and transforms_dict keys"

        sat_datamodules = {}

        for key, sat_data_dir in self.data_dir_dict.items():
            if key == "msg_cloudsat":
                datamodule_class = CloudsatMSGDataModule
            elif key == "goes_cloudsat":
                datamodule_class = CloudsatGOESDataModule
            elif key == "himawari_cloudsat":
                datamodule_class = CloudsatHIMAWARIDataModule
            else:
                raise NotImplementedError(f"Unknown satellite: {key}")

            sat_datamodules[key] = datamodule_class(
                data_dir=sat_data_dir,
                splits_dict=splits_dict,
                transforms=transforms_dict[key] if transforms_dict else None,
                ext=ext,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                load_overpass_mask=load_overpass_mask,
                load_zenith=load_zenith,
                load_solar=load_solar,
                cloudsat_variables=cloudsat_variables,
                file_number=file_number,
                filter_clear_sky=filter_clear_sky,
            )

        self.train_dataloaders = {
            key: sat_datamodules[key].train_dataloader()
            for key in sat_datamodules.keys()
        }

        self.test_dataloaders = {
            key: sat_datamodules[key].test_dataloader()
            for key in sat_datamodules.keys()
        }

        self.val_dataloaders = {
            key: sat_datamodules[key].val_dataloader() for key in sat_datamodules.keys()
        }

    def train_dataloader(self):
        """ Return the combined training dataloader across all CloudSat-paired satellites.

            Returns
            -------
            OneSatellitePerBatchDataLoader. Training dataloader sampling one satellite per batch.
        """
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.train_dataloaders, satellites=self.satellites, use_weights=self.weight_satellites
        )

    def test_dataloader(self):
        """ Return the combined test dataloader across all CloudSat-paired satellites.

            Returns
            -------
            OneSatellitePerBatchDataLoader. Test dataloader sampling one satellite per batch.
        """
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.test_dataloaders, satellites=self.satellites, use_weights=self.weight_satellites
        )

    def val_dataloader(self):
        """ Return the combined validation dataloader across all CloudSat-paired satellites.

            Returns
            -------
            OneSatellitePerBatchDataLoader. Validation dataloader sampling one satellite per batch.
        """
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.val_dataloaders, satellites=self.satellites, use_weights=self.weight_satellites
        )

