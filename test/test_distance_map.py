#%%
from calendar import c
from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO
import numpy as np
from omegaconf import OmegaConf
import torch
from get_model.dataset.zarr_dataset import InferenceDataset, PretrainDataset, get_hic_from_idx
from get_model.utils import rename_state_dict
config = OmegaConf.load('/home/xf2217/Projects/get_model/get_model/config/model/DistanceContactMap.yaml')
import hydra
model = hydra.utils.instantiate(config)['model']
checkpoint = torch.load('/home/xf2217/Projects/get_model/DistanceMap/62ne9tb8/checkpoints/best.ckpt')
model.load_state_dict(rename_state_dict(checkpoint['state_dict'], {'model.': ''}))
# %%
import hicstraw
esc = hicstraw.HiCFile('/home/xf2217/Projects/geneformer_nat/data/H1_ESC.hic')
# %%
# Configuration for Gencode
gencode_config = {
    "assembly": "hg38",
    "version": 40,
    "gtf_dir": "/home/xf2217/Projects/caesar/data/"
}
# Configuration for the dataset
dataset_config = {
    "zarr_dirs": ["/home/xf2217/Projects/get_data/encode_hg38atac_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/home/xf2217/Projects/get_data/hg38.zarr"},
    "genome_motif_zarr": "/home/xf2217/Projects/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "peaks_q0.05_fetal_joint_tissue_open_exp",
    "leave_out_chromosomes": None,
    "use_insulation": True,
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 900,
    "keep_celltypes": "k562.encode_hg38atac.ENCFF128WZG.max",
    "center_expand_target": 0,
    "random_shift_peak": 0,
    "peak_count_filter": 10,
    'mask_ratio': 0,
    "padding": 0,
    "hic_path": "/home/xf2217/Projects/geneformer_nat/data/H1_ESC.hic"
}
# %%
hg38 = DenseZarrIO('/home/xf2217/Projects/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)
# %%
dataset = PretrainDataset(is_train=False, sequence_obj={'hg38':hg38}, **dataset_config)
#%%
dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=['HBG1', 'HBE1'])
# %%
from get_model.dataset.collate import get_rev_collate_fn
dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=get_rev_collate_fn)
#%%
d = next(iter(dl))
# %%
d['peak_coord'] = d['celltype_peaks']
# %%
pred = model(model.get_input(d)['distance_map'])

distance = model.get_input(d)['distance_map'].squeeze().numpy()
# %%
import pandas as pd
from scipy.stats import pearsonr
df = pd.DataFrame({'pred': pred.detach().squeeze().numpy().flatten(), 'hic': d['hic_matrix'].squeeze().numpy().flatten(), 'distance': distance.flatten()})
#%%
from tqdm import tqdm
for i in tqdm(range(100)):
    d = next(iter(dl))
    d['peak_coord'] = d['celltype_peaks']
    pred = model(model.get_input(d)['distance_map'])
    distance = model.get_input(d)['distance_map'].squeeze().numpy()
    df = pd.concat([df, pd.DataFrame({'pred': pred.detach().squeeze().numpy().flatten(), 'hic': d['hic_matrix'].squeeze().numpy().flatten(), 'distance': distance.flatten()})])
# %%
df.shape
# %%
# df.plot.scatter(x='hic', y='pred', s=1)
# %%
df = df.query('distance<=5 & distance>=4')
#%%
df['distance_bin'] = pd.qcut(df['distance'], 50)
# remove bin with less than 10 points
#%%
df.distance_bin.value_counts()
# %%
corr_stratified = df.groupby('distance_bin').apply(lambda x: (pearsonr(x['pred'], x['hic'])[0], x['hic'].mean(), x['pred'].mean(), x.shape[0], 10**(x['distance'].min()-3), 10**(x['distance'].max()-3)))
# %%
corr_stratified = pd.DataFrame(corr_stratified.tolist(), columns=['corr', 'hic_mean', 'pred_mean', 'count', 'distance_min', 'distance_max'])
# %%
corr_stratified
# %%
corr_stratified.plot.scatter(x='distance_min', y='corr', ylim=(0, 1))
# %%
import seaborn as sns
sns.heatmap(pred.detach().squeeze().numpy() * (distance>5))
# %%
sns.heatmap(d['hic_matrix'].squeeze().numpy() * (distance>5))

# %%
def func(x):
    a,b,c = 74123.56198123861, 168568.99474605062, 0.21718070073422405
    return b / (a + np.array(x, dtype=np.float64))  + c

# %%
distance_expectation = func(10**(distance)-1)
# %%
distance_expectation
# %%
atac = d['atpm'].squeeze().numpy()
# outer product
atac = np.sqrt(atac[:, np.newaxis] * atac[np.newaxis, :])
# %%
# lower with hic and higher triangle with pred
lower = (d['hic_matrix'].squeeze().numpy()) * atac
upper = (pred.detach().squeeze().numpy()) * atac
lower[lower<0] = 0
upper[upper<0] = 0
upper=upper
m = np.zeros((lower.shape[0], lower.shape[1]))
m[np.tril_indices_from(m, -1)] = lower[np.tril_indices_from(lower, -1)]
m[np.triu_indices_from(m, 1)] = upper[np.triu_indices_from(upper, 1)]
# diagnal to 1
# m[np.diag_indices_from(m)] = 1
g = sns.heatmap(m, square=True, vmax=0.1)


# %%
lower_expectation = lower - distance_expectation*atac
upper_expectation = upper - distance_expectation*atac
sns.scatterplot(x=upper.flatten()[(lower_expectation.flatten()>0) & (upper_expectation.flatten()>0)], y=(distance_expectation*atac).flatten()[(lower_expectation.flatten()>0) & (upper_expectation.flatten()>0)], s=2, hue=distance.flatten()[(lower_expectation.flatten()>0) & (upper_expectation.flatten()>0)])
# %%
def KR_normalization(matrix):
    """
    KR normalization for Hi-C matrix, which makes the sum of each row and column to be 1
    """
    sum = np.sum(matrix, axis=0)
    sum = np.sqrt(sum[:, np.newaxis] * sum[np.newaxis, :])
    return matrix / sum

# %%
sns.scatterplot(x=upper.flatten(), y = (distance_expectation*atac).flatten(), s=2, hue=distance.flatten())
# %%
upper
# %%
