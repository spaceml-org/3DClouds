from __future__ import annotations

import lightning.pytorch as pl
from loguru import logger

from src import utils
from collections import defaultdict
import torch
import torch.nn as nn


class ModelFromHydraRun(pl.LightningModule):
    """ Lightning wrapper that loads a model from a saved Hydra run checkpoint. """

    def __init__(
        self,
        hydra_run_path,
        model_expression=None,
        freeze=False,
        unfreeze=False,
        loading_from_state_dict=True,
        enable_loading_weights: bool = True,
        model_setvars={},
    ):
        """ Initialize ModelFromHydraRun.

            Parameters
            ----------
            hydra_run_path : str. Path to the Hydra run directory; see utils.load_ckpt_from_hydra_run.
            model_expression : str or None. Python expression evaluated on the loaded model object
                to select a sub-module (e.g. 'backbone') (optional).
            freeze : bool. If True, freeze all model parameters after loading (optional).
            unfreeze : bool. If True, unfreeze all model parameters after loading (optional).
            loading_from_state_dict : bool. If True, load weights from a state-dict checkpoint (optional).
            enable_loading_weights : bool. If True, actually load the checkpoint weights (optional).
            model_setvars : dict. Mapping of attribute expressions to values set on the model
                after loading (e.g. {'encoder.mask_ratio': 0.0}) (optional).

            Returns
            -------
            None.
        """
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        self.hydra_run_path = hydra_run_path
        self.model_expression = model_expression
        self.freeze = freeze
        self.unfreeze = unfreeze

        # create model
        self.model = utils.load_ckpt_from_hydra_run(
            hydra_run_path,
            loading_from_state_dict=loading_from_state_dict,
            enable_loading_weights=enable_loading_weights,
        )
        if model_expression is not None:
            self.model = eval(f"self.model.{model_expression}")

        # freeze
        if self.freeze:
            try:
                self.model.freeze()
            except:
                logger.info("Model is not PLModule, freezing manually")
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()  # Ensure evaluation mode manually

            logger.info("-------------------------")
            logger.info(f"freezing {freeze}")
            logger.info("-------------------------")

        # unfreeze
        if self.unfreeze:
            try:
                self.model.unfreeze()
            except:
                logger.info("Model is not PLModule, freezing manually")
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.train()  # Ensure evaluation mode manually

            logger.info("-------------------------")
            logger.info(f"unfreezing {unfreeze}")
            logger.info("-------------------------")

        for k, v in model_setvars.items():
            logger.info(f"---- model setting {k} to {v}")
            exec(f"self.model.{k} = {v}")

        if "output_dim" in dir(self.model):
            self.output_dim = self.model.output_dim

    def forward(self, x):
        """ Forward pass delegated to the wrapped model.

            Parameters
            ----------
            x : any. Input passed directly to the underlying model's forward method.

            Returns
            -------
            any. Output of the underlying model.
        """
        return self.model(x)



def nontrainable_breakdown(model: nn.Module, topn=30):
    """ Log a breakdown of non-trainable parameter counts by owning module.

        Parameters
        ----------
        model : nn.Module. The model to inspect.
        topn : int. Maximum number of modules to report, sorted by parameter count (optional).

        Returns
        -------
        None.
    """
    # Map each parameter to the *immediate* owning module for precise attribution
    owning = {}
    for mod_name, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            owning[id(p)] = mod_name or "model"

    counts = defaultdict(int)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            key = owning.get(id(p), name.split('.')[0])
            counts[key] += p.numel()

    total = sum(counts.values())
    logger.info(f"Non-trainable total: {total:,}")
    for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:topn]:
        logger.info(f"{k}: {v:,}")

