#%%
import logging
import re
from functools import partial

import hydra
import lightning as L
import pandas as pd
import seaborn as sns
import torch
import torch.utils.data
import zarr
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.plugins import MixedPrecision
from matplotlib import pyplot as plt
from minlora import LoRAParametrization
from minlora.model import add_lora_by_name
from omegaconf import MISSING, DictConfig, OmegaConf
from tqdm import tqdm

import wandb
from get_model.config.config import *
from get_model.dataset.collate import everything_collate
from get_model.dataset.zarr_dataset import (
    EverythingDataset, InferenceEverythingDataset,
    InferenceReferenceRegionDataset,
    PerturbationInferenceReferenceRegionDataset, ReferenceRegionDataset,
    ReferenceRegionMotif, ReferenceRegionMotifConfig)
from get_model.model.model_refactored import *
from get_model.model.modules import *
from get_model.optim import LayerDecayValueAssigner, create_optimizer
from get_model.run import GETDataModule, LitModel, get_insulation_overlap
from get_model.utils import (cosine_scheduler, extract_state_dict,
                             load_checkpoint, load_state_dict,
                             recursive_detach, recursive_numpy,
                             recursive_save_to_zarr, rename_state_dict)

np.bool = np.bool_
def run_downstream(cfg: DictConfig):
    torch.set_float32_matmul_precision('medium')
    # if cfg.finetune.checkpoint is not None:
    # model = LitModel.load_from_checkpoint(cfg.finetune.checkpoint)
    # else:
    model = LitModel(cfg)
    # move the model to the gpu
    model.to('cuda')
    dm = GETDataModule(cfg)
    model.dm = dm
    if cfg.machine.num_devices > 0:
        strategy = 'auto'
        accelerator = 'gpu'
        device = cfg.machine.num_devices
        if cfg.machine.num_devices > 1:
            strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'
        accelerator = 'cpu'
        device = 'auto'
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        num_sanity_val_steps=10,
        strategy=strategy,
        devices=device,
        # plugins=[MixedPrecision(precision='16-mixed', device="cuda")],
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.clip_grad,
        log_every_n_steps=100,
        deterministic=True,
        default_root_dir=cfg.machine.output_dir,
    )
    return run_ppif_task(trainer, model)


def run_ppif_task(trainer: L.Trainer, lm: LitModel, output_key='atpm'):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    mutation = pd.read_csv(
        lm.cfg.task.mutations, sep='\t')
    n_mutation = mutation.shape[0]
    n_peaks_upper_bound = lm.cfg.dataset.n_peaks_upper_bound
    result = []
    # setup dataset_predict
    lm.dm.setup(stage='predict')
    with torch.no_grad():
        lm.to("cuda")
        for i, batch in tqdm(enumerate(lm.dm.predict_dataloader()), total=len(lm.dm.predict_dataloader())):
            batch = lm.transfer_batch_to_device(
                batch, lm.device, dataloader_idx=0)
            out = lm.predict_step(batch, i)
            result.append(out)
    return mutation, result, lm, trainer
    # pred_wt = [r['pred_wt'][output_key] for r in result]
    # pred_mut = [r['pred_mut'][output_key] for r in result]
    # n_celltypes = lm.dm.dataset_predict.inference_dataset.datapool.n_celltypes
    # pred_wt = torch.cat(pred_wt, dim=0).reshape(
    #     n_celltypes, n_mutation, n_peaks_upper_bound)[0, :, n_peaks_upper_bound//2]
    # pred_mut = torch.cat(pred_mut, dim=0).reshape(
    #     n_celltypes, n_mutation, n_peaks_upper_bound)[0, :, n_peaks_upper_bound//2]
    # pred_change = pred_mut-pred_wt
    # mutation['pred_change'] = pred_change.detach().cpu().numpy()
    # y = mutation.query('`corrected p value`<=0.05').query('Screen.str.contains("Pro")').query('Screen.str.contains("Tiling")')[
    #     '% change to PPIF expression'].values
    # x = mutation.query('`corrected p value`<=0.05').query('Screen.str.contains("Pro")').query(
    #     'Screen.str.contains("Tiling")')['pred_change'].values
    # pearson = np.corrcoef(x, y)[0, 1]
    # r2 = r2_score(y, x)
    # spearman = spearmanr(x, y)[0]
    # slope = LinearRegression().fit(x.reshape(-1, 1), y).coef_[0]
    # # save a scatterplot
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.scatterplot(x=x, y=y)
    # plt.xlabel('Predicted change in PPIF expression')
    # plt.ylabel('Observed change in PPIF expression')
    # plt.savefig(
    #     f'{lm.cfg.machine.output_dir}/ppif_scatterplot.png')
    # return {
    #     'ppif_pearson': pearson,
    #     'ppif_spearman': spearman,
    #     'ppif_r2': r2,
    #     'ppif_slope': slope
    # }
    # return mutation, result, lm, trainer

# %%
def load_config(config_name):
    # Initialize Hydra to load the configuration
    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../get_model/config", version_base="1.3")
    cfg = hydra.compose(config_name=config_name)
    return cfg

# %%
config_name = "eval_k562_fetal_nucleotide_region_finetune_atac"
config_name = 'chrombpnet_k562'
cfg = load_config(config_name)
# %%
cfg.task.mutations = '/home/xf2217/Projects/get_model/evals/cagi/cagi_k562_pklr_hg38.tsv'
cfg.task.gene_list = 'PKLR,'
cfg.task.test_mode = 'perturb'
cfg.finetune.use_lora = True
cfg.finetune.strict=True
cfg.dataset.leave_out_chromosomes=None
cfg.finetune.checkpoint = '/home/xf2217/Projects/get_model/nucleotide_region_finetune_atac/3dcgb0to/checkpoints/last.ckpt'

#%%
cfg.finetune.model_key ='model.'
cfg.finetune.use_lora = False
cfg.dataset.random_shift_peak = 0
cfg.finetune.checkpoint = '/home/xf2217/Projects/get_model/k562_chrombpnet/jhxywyli/checkpoints/last.ckpt'
# %%
mutation, result, lm, trainer = run_downstream(cfg)
# %%
output_key='aprofile'
pred_wt = [r['pred_wt'][output_key] for r in result]
pred_mut = [r['pred_mut'][output_key] for r in result]
n_celltypes = lm.dm.dataset_predict.inference_dataset.datapool.n_celltypes
pred_wt = torch.cat(pred_wt, dim=0)
pred_mut = torch.cat(pred_mut, dim=0)
pred_change = (pred_mut-pred_wt)
mutation['pred_change_aprofile'] = pred_change.sum(dim=2).detach().cpu().numpy()
#%%
output_key ='atpm'
n_mutation = 1409
n_peaks_upper_bound = lm.cfg.dataset.n_peaks_upper_bound

pred_wt = [r['pred_wt'][output_key] for r in result]
pred_mut = [r['pred_mut'][output_key] for r in result]
n_celltypes = lm.dm.dataset_predict.inference_dataset.datapool.n_celltypes
pred_wt = torch.cat(pred_wt, dim=0).reshape(
    n_celltypes, n_mutation, n_peaks_upper_bound)[0, :, n_peaks_upper_bound//2]
pred_mut = torch.cat(pred_mut, dim=0).reshape(
    n_celltypes, n_mutation, n_peaks_upper_bound)[0, :, n_peaks_upper_bound//2]
pred_change = pred_mut-pred_wt
mutation['pred_change'] = pred_change.detach().cpu().numpy()


# %%
sns.scatterplot(y='Value', x='pred_change_aprofile', data=mutation, hue='Confidence', s=1)
# %%
# correlation
mutation[[ 'pred_change_aprofile', 'Value']].corr('spearman')
# %%
sns.scatterplot(x=result[0]['pred_wt']['aprofile'].cpu().numpy()[0,:].flatten(), y=result[0]['obs_wt']['aprofile'].cpu().numpy()[0,:].flatten())

# %%
result[1]['obs_wt']['aprofile'].cpu().numpy().shape
# %%
sns.scatterplot(x=result[0]['pred_wt']['atpm'].cpu().numpy()[:,100].flatten(), y=result[0]['obs_mut']['atpm'].cpu().numpy()[:,100].flatten())
# %%
sns.lineplot(x=range(len(result[0]['obs_wt']['aprofile'].cpu().numpy()[0,0,:])), y=result[0]['obs_mut']['aprofile'].cpu().numpy()[0,0,:])
sns.lineplot(x=range(len(result[0]['obs_wt']['aprofile'].cpu().numpy()[0,0,:])), y=result[0]['pred_wt']['aprofile'].cpu().numpy()[0,0,:])
sns.lineplot(x=range(len(result[0]['obs_wt']['aprofile'].cpu().numpy()[0,0,:])), y=result[0]['pred_mut']['aprofile'].cpu().numpy()[0,0,:])

sns.lineplot(x=range(len(result[0]['obs_wt']['aprofile'].cpu().numpy()[0,0,:])), y=result[0]['pred_mut']['aprofile'].cpu().numpy()[0,0,:]-result[0]['pred_wt']['aprofile'].cpu().numpy()[0,0,:])

# %%
(result[0]['obs_wt']['aprofile']-result[0]['pred_wt']['aprofile']).mean()
# %%
lm.dm.dataset_predict.perturbations_gene_overlap

# %%
lm.dm.dataset_predict[0]
# %%
fig, ax = plt.subplots(10, 1, figsize=(10, 20))
for j, i in enumerate(pred_change[0:10]):
    sns.lineplot(x=range(len(i[0])), y=i[0].cpu().numpy(), ax=ax[j])
# %%
mutation
# %%
peak = lm.dm.dataset_predict.inference_dataset.datapool.peaks_dict['k562.encode_hg38atac.ENCFF257HEE.max'].query('Count>10')
# %%
peak['length'] = peak['End']-peak['Start']
peak.plot(x='aTPM', y='length', kind='scatter')
# %%
from caesar.io.zarr_io import CelltypeDenseZarrIO
c = CelltypeDenseZarrIO('/home/xf2217/Projects/get_data/encode_hg38atac_dense.zarr/')
# %%
peak = c.get_peaks('k562.encode_hg38atac.ENCFF257HEE.max', 'peaks_q0.05_fetal_joint_tissue_open_exp')
peak['length'] = peak['End']-peak['Start']
peak.query('length<5000').plot(x='aTPM', y='length', kind='scatter',s=1, alpha=0.3)
# %%
peak[['length', 'aTPM']].corr('spearman')
# %%
lm.model
# %%
dataset_track = np.array(lm.dm.dataset_predict.inference_dataset[0]['sample_track'].todense()).flatten()
conv =50
dataset_track = np.convolve(
            np.array(dataset_track).reshape(-1), np.ones(conv)/conv, mode='same')
sns.lineplot(x=range(len(dataset_track)), y=dataset_track)
dataset_track = np.log10(dataset_track+1)
# %%
# center 1000bp of 2114
sns.lineplot(x=range(1000), y=dataset_track[556:1556])
# %%
sns.lineplot(x=range(len(result[0]['obs_wt']['aprofile'].cpu().numpy()[0,0,:])), y=result[0]['obs_mut']['aprofile'].cpu().numpy()[0,0,:]/result[0]['obs_mut']['aprofile'].cpu().numpy()[0,0,:].max())
sns.lineplot(x=range(len(result[0]['obs_wt']['aprofile'].cpu().numpy()[0,0,:])), y=result[0]['pred_wt']['aprofile'].cpu().numpy()[0,0,:]/result[0]['pred_wt']['aprofile'].cpu().numpy()[0,0,:].max())
sns.lineplot(x=range(len(result[0]['obs_wt']['aprofile'].cpu().numpy()[0,0,:])), y=result[0]['pred_mut']['aprofile'].cpu().numpy()[0,0,:]/result[0]['pred_mut']['aprofile'].cpu().numpy()[0,0,:].max())

# sns.lineplot(x=range(len(result[0]['obs_wt']['aprofile'].cpu().numpy()[0,0,:])), y=result[0]['pred_mut']['aprofile'].cpu().numpy()[0,0,:]-result[0]['pred_wt']['aprofile'].cpu().numpy()[0,0,:])
sns.lineplot(x=range(1000), y=dataset_track[556:1556]/dataset_track[556:1556].max())
# %%
