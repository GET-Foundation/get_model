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
<<<<<<< HEAD
        # Prepare data for plotting
        plot_data = pd.DataFrame({
            '% change to PPIF expression': self.mutations['% change to PPIF expression'],
            'ap_fc': self.mut_predictions - self.wt_predictions
        })

        # Create plot
        sns.scatterplot(data=plot_data.query(
            'Screen.str.contains("Pro")'), x='% change to PPIF expression', y='ap_fc')
        plt.savefig('mutation_impact.png')
=======
        pass


def run_ppif_task(trainer: L.Trainer, lm: LitModel):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    mutation = pd.read_csv(
        '/home/xf2217/Projects/get_data/prepared_data.tsv', sep='\t')
    n_mutation = mutation.shape[0]
    n_peaks_upper_bound = lm.cfg.dataset.n_peaks_upper_bound
    result = []
    # setup dataset_predict
    lm.dm.setup(stage='predict')
    for i, batch in tqdm(enumerate(lm.dm.predict_dataloader())):
        batch = lm.transfer_batch_to_device(batch, lm.device, dataloader_idx=0)
        out = lm.predict_step(batch, i)
        result.append(out)
    pred_wt = [r['pred_wt']['exp'] for r in result]
    pred_mut = [r['pred_mut']['exp'] for r in result]
    n_celltypes = trainer.datamodule.dataset_predict.inference_dataset.datapool.n_celltypes
    pred_wt = torch.cat(pred_wt, dim=0).reshape(
        n_celltypes, n_mutation, n_peaks_upper_bound, 2)[0, :, n_peaks_upper_bound//2, 0]
    pred_mut = torch.cat(pred_mut, dim=0).reshape(
        n_celltypes, n_mutation, n_peaks_upper_bound, 2)[0, :, n_peaks_upper_bound//2, 0]
    pred_change = ((10**pred_mut-1) - (10**pred_wt - 1)) / \
        (10**pred_wt - 1) * 100
    mutation['pred_change'] = pred_change.cpu().numpy()
    y = mutation['% change to PPIF expression'].values
    x = mutation['pred_change'].values
    pearson = np.corrcoef(x, y)[0, 1]
    r2 = r2_score(y, x)
    spearman = spearmanr(x, y)[0]
    slope = LinearRegression().fit(x.reshape(-1, 1), y).coef_[0]
    return {
        'ppif_pearson': pearson,
        'ppif_spearman': spearman,
        'ppif_r2': r2,
        'ppif_slope': slope
    }
>>>>>>> ae2ddcc690feffd6c85121730329d5aa24d9adad
