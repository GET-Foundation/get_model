from dataclasses import dataclass

import pandas as pd
import torch
import torch.utils.data
from gcell.rna.gencode import Gencode
from omegaconf import MISSING

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import InferenceDataset


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
            dataset, batch_size=1, collate_fn=get_rev_collate_fn
        )
        return dataloader

    def mut_dataloader(self, lm):
        dataset = InferenceDataset(
            **lm.dataset_config, gencode_obj=lm.gencode, mut=self.mutations
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=get_rev_collate_fn
        )
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
