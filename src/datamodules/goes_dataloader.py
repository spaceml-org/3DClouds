from __future__ import annotations

import autoroot  # required to load from src
from lightning.pytorch import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader

from src.datamodules.constants import SPLITS_DICT
from src.datamodules.goes_dataset import GOESDataset
from src.utils import get_list_filenames, get_split


class GOESDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir,
        splits_dict=SPLITS_DICT,
        transforms=None,
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
        self.save_hyperparameters(logger=False)
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        self.data_dir = data_dir
        self.splits_dict = splits_dict
        self.ext = ext
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_overpass_mask = return_overpass_mask
        self.load_zenith = load_zenith
        self.load_solar = load_solar
        self.patch_size = patch_size
        self.center_crop = center_crop
        self.radius = radius

        # get all filenames in data_dir
        filenames = get_list_filenames(data_path=self.data_dir, ext=self.ext)

        logger.info(f"Found {len(filenames)} files in {self.data_dir}")

        # split filenames based on train/test/val criteria
        train_files = get_split(filenames, splits_dict["train"])
        test_files = get_split(filenames, splits_dict["test"])
        val_files = get_split(filenames, splits_dict["val"])

        self.train_dataset = GOESDataset(
            data_filenames=train_files,
            transforms=self.transforms,
            return_overpass_mask=self.return_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            patch_size=self.patch_size,
            center_crop=self.center_crop,
            radius=self.radius,
        )

        self.test_dataset = GOESDataset(
            data_filenames=test_files,
            transforms=self.transforms,
            return_overpass_mask=self.return_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            patch_size=self.patch_size,
            center_crop=self.center_crop,
            radius=self.radius,
        )

        self.val_dataset = GOESDataset(
            data_filenames=val_files,
            transforms=self.transforms,
            return_overpass_mask=self.return_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            patch_size=self.patch_size,
            center_crop=self.center_crop,
            radius=self.radius,
        )

        logger.info("GOES DataModule initialized ...")
        logger.info(f"Length of train dataset: {len(self.train_dataset)}")
        logger.info(f"Length of test dataset: {len(self.test_dataset)}")
        logger.info(f"Length of val dataset: {len(self.val_dataset)}")

    def prepare_data(self):
        self.train_dataset.prepare_data()
        self.test_dataset.prepare_data()
        self.val_dataset.prepare_data()

    def setup(self, stage):
        self.train_dataset.setup(stage)
        self.test_dataset.setup(stage)
        self.val_dataset.setup(stage)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )
