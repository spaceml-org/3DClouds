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
    def __init__(
        self,
        sat_dataloaders: dict,
        satellites: list[str],
        use_weights: bool = False,
    ):
        self.sat_dataloaders = sat_dataloaders
        self.satellites = satellites
        self.iters = {sat: iter(dl) for sat, dl in sat_dataloaders.items()}
        lengths = [len(dl) for dl in sat_dataloaders.values()]
        self.length = sum(lengths)
        self.weights = lengths if use_weights else None

    def __iter__(self):
        return self

    def __next__(self):
        sat = random.choices(self.satellites, weights=self.weights)[0]

        try:
            return next(self.iters[sat])
        except StopIteration:
            # Restart exhausted loader
            # NOTE it should still finish iterating once __len__ is reached
            logger.info(f"Restarting iterator for {sat} dataloader")
            self.iters[sat] = iter(self.sat_dataloaders[sat])
            return next(self.iters[sat])

        # NOTE if restarting iterator is unacceptable, this option should work:
        # if satellites is None:
        #     sat = random.choice(self.satellites)
        # elif satellites == []:
        #     raise StopIteration("No more satellites to iterate over")
        # else:
        #     sat = random.choice(satellites)
        # try:
        #     return next(self.iters[sat])
        # except StopIteration:
        #     new_satellites = [s for s in self.satellites if s != sat]
        #     return self.__next__(satellites=new_satellites)

    def __len__(self):
        return self.length


class MultiDataModule(LightningDataModule):
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
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.train_dataloaders, satellites=self.satellites
        )

    def test_dataloader(self):
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.test_dataloaders, satellites=self.satellites
        )

    def val_dataloader(self):
        return OneSatellitePerBatchDataLoader(
            sat_dataloaders=self.val_dataloaders, satellites=self.satellites
        )

