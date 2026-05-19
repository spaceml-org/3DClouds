from __future__ import annotations

import autoroot  # required to load from src
import numpy as np
from lightning.pytorch import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader

from src.datamodules.cloudsat_msg_dataset import CloudsatMSGDataset
from src.datamodules.constants import SPLITS_DICT
from src.datamodules.utils import filter_files
from src.utils import get_list_filenames, get_split


class CloudsatMSGDataModule(LightningDataModule):
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
        load_overpass_mask: bool = False,  # defaults to False for pre-training
        load_zenith: bool = True,
        load_solar: bool = True,
        cloudsat_variables: list[str] = ["Radar_Reflectivity"],
        file_number: int = None,
        filter_clear_sky: dict | None = None,
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
        self.load_overpass_mask = load_overpass_mask
        self.load_zenith = load_zenith
        self.load_solar = load_solar
        self.cloudsat_variables = cloudsat_variables
        self.file_number = file_number
        self.filter_clear_sky = filter_clear_sky

        # get all filenames in data_dir
        filenames = get_list_filenames(data_path=self.data_dir, ext=self.ext)
        logger.info(f"Found {len(filenames)} files in {self.data_dir}")

        if self.filter_clear_sky is not None:
            filenames = filter_files(
                filenames=filenames,
                filter_clear_sky=self.filter_clear_sky,
                satellite="msg",
            )
            logger.info(f"Found {len(filenames)} files after clear sky filtering...")

        # if file_number is set, randomly select that many files
        if self.file_number is not None and self.file_number < len(filenames):
            filenames = np.random.choice(
                filenames, size=self.file_number, replace=False
            )
            logger.info(f"Selected {len(filenames)} files")

        # split filenames based on train/test/val criteria
        train_files = get_split(filenames, splits_dict["train"])
        test_files = get_split(filenames, splits_dict["test"])
        val_files = get_split(filenames, splits_dict["val"])

        self.train_dataset = CloudsatMSGDataset(
            data_filenames=train_files,
            transforms=self.transforms,
            load_overpass_mask=self.load_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            cloudsat_variables=self.cloudsat_variables,
        )

        self.test_dataset = CloudsatMSGDataset(
            data_filenames=test_files,
            transforms=self.transforms,
            load_overpass_mask=self.load_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            cloudsat_variables=self.cloudsat_variables,
        )

        self.val_dataset = CloudsatMSGDataset(
            data_filenames=val_files,
            transforms=self.transforms,
            load_overpass_mask=self.load_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            cloudsat_variables=self.cloudsat_variables,
        )

        logger.info("CLOUDSAT-MSG DataModule initialized ...")
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
            # collate_fn=safe_tensor_collate,
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
            # collate_fn=safe_tensor_collate,
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
            # collate_fn=safe_tensor_collate,
        )

