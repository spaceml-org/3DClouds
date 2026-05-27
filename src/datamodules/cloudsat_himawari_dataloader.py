from __future__ import annotations

import autoroot  # required to load from src
import numpy as np
from lightning.pytorch import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader

from src.datamodules.cloudsat_himawari_dataset import CloudsatHIMAWARIDataset
from src.datamodules.constants import SPLITS_DICT
from src.datamodules.utils import filter_files
from src.utils import get_list_filenames, get_split


class CloudsatHIMAWARIDataModule(LightningDataModule):
    """ LightningDataModule for CloudSat-paired Himawari satellite data. """

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
        """ Initialize CloudsatHIMAWARIDataModule.

            Parameters
            ----------
            data_dir : str. Path to the directory containing CloudSat-Himawari NetCDF files.
            splits_dict : dict. Dictionary specifying train/test/val split criteria (optional).
            transforms : callable | None. Transform to apply to each sample (optional).
            ext : str. File extension to search for in data_dir (optional).
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

            Returns
            -------
            None.
        """
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
                satellite="himawari",
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

        self.train_dataset = CloudsatHIMAWARIDataset(
            data_filenames=train_files,
            transforms=self.transforms,
            load_overpass_mask=self.load_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            cloudsat_variables=self.cloudsat_variables,
        )

        self.test_dataset = CloudsatHIMAWARIDataset(
            data_filenames=test_files,
            transforms=self.transforms,
            load_overpass_mask=self.load_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            cloudsat_variables=self.cloudsat_variables,
        )

        self.val_dataset = CloudsatHIMAWARIDataset(
            data_filenames=val_files,
            transforms=self.transforms,
            load_overpass_mask=self.load_overpass_mask,
            load_zenith=self.load_zenith,
            load_solar=self.load_solar,
            cloudsat_variables=self.cloudsat_variables,
        )

        logger.info("CLOUDSAT-HIMAWARI DataModule initialized ...")
        logger.info(f"Length of train dataset: {len(self.train_dataset)}")
        logger.info(f"Length of test dataset: {len(self.test_dataset)}")
        logger.info(f"Length of val dataset: {len(self.val_dataset)}")

    def prepare_data(self):
        """ Delegate prepare_data to all sub-datasets. """
        self.train_dataset.prepare_data()
        self.test_dataset.prepare_data()
        self.val_dataset.prepare_data()

    def setup(self, stage):
        """ Delegate setup to all sub-datasets.

            Parameters
            ----------
            stage : str. One of 'fit', 'validate', 'test', or 'predict'.

            Returns
            -------
            None.
        """
        self.train_dataset.setup(stage)
        self.test_dataset.setup(stage)
        self.val_dataset.setup(stage)

    def train_dataloader(self):
        """ Return a shuffled DataLoader over the training split.

            Returns
            -------
            DataLoader. Training DataLoader with shuffle=True.
        """
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
        """ Return a DataLoader over the validation split.

            Returns
            -------
            DataLoader. Validation DataLoader with shuffle=False.
        """
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
        """ Return a DataLoader over the test split.

            Returns
            -------
            DataLoader. Test DataLoader with shuffle=False.
        """
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

