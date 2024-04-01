
import lightning as L
import pandas as pd
import torch
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
        mutations = pd.read_csv(self.mutation_file)
        self.mutations = mutations

    def wt_dataloader(self):
        pass

    def mut_dataloader(self):
        pass

    def predict(self, lm: LitModel):
        wt_predictions = []
        mut_predictions = []
        for batch in self.wt_dataloader:
            wt_predictions.append(lm.model(batch))

        for batch in self.mut_dataloader:
            mut_predictions.append(lm.model(batch))

    def plot(self):
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
