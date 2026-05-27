from __future__ import annotations

import random
from typing import Callable, Optional

import autoroot  # required to load from src
from lightning.pytorch import LightningDataModule
from loguru import logger

from src.datamodules.constants import SPLITS_DICT
from src.datamodules.goes_dataloader import GOESDataModule
from src.datamodules.himawari_dataloader import HIMAWARIDataModule
from src.datamodules.msg_dataloader import MSGDataModule


class OneSatellitePerBatchDataLoader:
    """ Iterator that samples one batch from a randomly selected satellite dataloader per step. """

    def __init__(
        self,
        sat_dataloaders: dict,
        satellites: list[str],
        use_weights: bool = False,
    ):
        """ Initialize OneSatellitePerBatchDataLoader.

            Parameters
            ----------
            sat_dataloaders : dict. Mapping from satellite name to its DataLoader.
            satellites : list[str]. List of satellite names to sample from.
            use_weights : bool. If True, samples satellites proportionally to their dataset length (optional).

            Returns
            -------
            None.
        """
        self.sat_dataloaders = sat_dataloaders
        self.satellites = satellites
        self.iters = {sat: iter(dl) for sat, dl in sat_dataloaders.items()}
        lengths = [len(dl) for dl in sat_dataloaders.values()]
        self.length = sum(lengths)
        self.weights = lengths if use_weights else None

    def __iter__(self):
        """ Return the iterator object itself. """
        return self

    def __next__(self):
        """ Return the next batch from a randomly chosen satellite dataloader.

            Returns
            -------
            batch. Next batch from the selected satellite dataloader.
        """
        sat = random.choices(self.satellites, weights=self.weights)[0]

        try:
            return next(self.iters[sat])
        except StopIteration:
            # Restart exhausted loader
            # NOTE it should still finish iterating once __len__ is reached
            logger.info(f"Restarting iterator for {sat} dataloader")
            self.iters[sat] = iter(self.sat_dataloaders[sat])
            return next(self.iters[sat])

    def __len__(self):
        """ Return the total combined length of all satellite dataloaders.

            Returns
            -------
            int. Sum of lengths of all satellite dataloaders.
        """
        return self.length


class MultiDataModule(LightningDataModule):
    """ LightningDataModule combining MSG, GOES, and Himawari satellite data for multi-source training. """

    def __init__(
        self,
        data_dir_dict: dict[str, str],
        splits_dict=SPLITS_DICT,
        satellites: list[str] = ["msg", "goes", "himawari"],
        transforms_dict: dict[str, Callable] | None = None,
        ext: str = "nc",
        batch_size: int = 4,
        num_workers: int = 1,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        return_overpass_mask: bool = False,  # defaults to False for pre-training
        load_zenith: bool = True,
        load_solar: bool = True,
        patch_size: list = None,  # whether to crop the data to a smaller patch size (e.g. [128, 128])
        center_crop: bool = False,  # If True, will crop to the center of the image
        radius: int = 0,  # Radius for cropping, if center_crop is True
    ):
        """ Initialize MultiDataModule.

            Parameters
            ----------
            data_dir_dict : dict[str, str]. Mapping from satellite name to its data directory path.
            splits_dict : dict. Dictionary specifying train/test/val split criteria (optional).
            satellites : list[str]. List of satellite names to include; must match data_dir_dict keys (optional).
            transforms_dict : dict[str, Callable] | None. Per-satellite transform callables (optional).
            ext : str. File extension to search for in data directories (optional).
            batch_size : int. Number of samples per batch (optional).
            num_workers : int. Number of DataLoader worker processes (optional).
            pin_memory : bool. If True, pins tensors to memory for faster GPU transfer (optional).
            prefetch_factor : int. Number of batches to prefetch per worker (optional).
            return_overpass_mask : bool. If True, returns the CloudSat overpass mask (optional).
            load_zenith : bool. If True, loads zenith angle data (optional).
            load_solar : bool. If True, loads solar angle data (optional).
            patch_size : list. Crop size as [H, W]; pass a dict for per-satellite sizes (optional).
            center_crop : bool. If True, crops to the center of the image (optional).
            radius : int. Radius in pixels used when center_crop is True (optional).

            Returns
            -------
            None.
        """
        super().__init__()
        self.data_dir_dict = data_dir_dict
        self.satellites = satellites

        assert set(satellites) == set(
            data_dir_dict.keys()
        ), "Mismatch between satellites list and filenames_dict keys"
        if transforms_dict:
            assert set(satellites) == set(
                transforms_dict.keys()
            ), "Mismatch between satellites list and transforms_dict keys"

        sat_datamodules = {}

        for key, sat_data_dir in self.data_dir_dict.items():
            if key == "msg":
                datamodule_class = MSGDataModule
            elif key == "goes":
                datamodule_class = GOESDataModule
            elif key == "himawari":
                datamodule_class = HIMAWARIDataModule
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
                return_overpass_mask=return_overpass_mask,
                load_zenith=load_zenith,
                load_solar=load_solar,
                patch_size=patch_size[key] if type(patch_size) is dict else patch_size,
                center_crop=center_crop,
                radius=radius,
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
        """ Return the combined training dataloader across all satellites.

            Returns
            -------
            OneSatellitePerBatchDataLoader. Training dataloader sampling one satellite per batch.
        """
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.train_dataloaders, satellites=self.satellites
        )

    def test_dataloader(self):
        """ Return the combined test dataloader across all satellites.

            Returns
            -------
            OneSatellitePerBatchDataLoader. Test dataloader sampling one satellite per batch.
        """
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.test_dataloaders, satellites=self.satellites
        )

    def val_dataloader(self):
        """ Return the combined validation dataloader across all satellites.

            Returns
            -------
            OneSatellitePerBatchDataLoader. Validation dataloader sampling one satellite per batch.
        """
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.val_dataloaders, satellites=self.satellites
        )

