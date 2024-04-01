import logging

import lightning as L
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.utils.data
from caesar.io.zarr_io import DenseZarrIO
from caesar.io.gencode import Gencode
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from lightning.pytorch.utilities import grad_norm
from omegaconf import MISSING, DictConfig

from get_model.config.config import *
from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import DenseZarrIO, InferenceDataset, PretrainDataset
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import create_optimizer
from get_model.utils import cosine_scheduler, load_checkpoint, remove_keys
import seaborn as sns


from dataclasses import dataclass

import lightning as L
import pandas as pd
import torch
from omegaconf import MISSING
from tqdm import tqdm

from get_model.run import LitModel


@dataclass
class BaseTaskConfig:
    model: str = MISSING
    metadata: str = MISSING


@dataclass
class MutationTaskConfig(BaseTaskConfig):
    mutation_file: str = MISSING


@dataclass
class PeakInactivationTaskConfig(BaseTaskConfig):
    peak_file: str = MISSING


class BaseTask:
    def __init__(self, cfg: BaseTaskConfig):
        self.cfg = cfg
        self.gencode = Gencode(cfg.assembly)
        self.load_metadata()

    def load_metadata(self):
        metadata = pd.read_csv(self.cfg.metadata)
        self.metadata = metadata

    def predict(self):
        pass

    def plot(self):
        pass


class MutationTask(BaseTask):
    def __init__(self, cfg: MutationTaskConfig):
        super().__init__(cfg)
        self.load_mutation_file()

    def load_mutation_file(self):
        mutations = pd.read_csv(self.cfg.mutation_file)
        self.mutations = mutations

    def wt_dataloader(self, lm):
        dataset = InferenceDataset(**lm.dataset_config)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=get_rev_collate_fn)
        return dataloader

    def mut_dataloader(self, lm):
        dataset = InferenceDataset(
            **lm.dataset_config, gencode_obj=lm.gencode, mut=self.mutations)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=get_rev_collate_fn)
        return dataloader

    def predict(self, lm):
        lm.model.eval()
        wt_predictions = []
        mut_predictions = []

        for batch in self.wt_dataloader(lm):
            with torch.no_grad():
                wt_predictions.append(lm.model(batch))

        for batch in self.mut_dataloader(lm):
            with torch.no_grad():
                mut_predictions.append(lm.model(batch))

        self.wt_predictions = wt_predictions
        self.mut_predictions = mut_predictions

    def plot(self):
        pass

