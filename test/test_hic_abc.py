# %%
import seaborn as sns
import torch.utils
from get_model.run_ref_region import *
import random

from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO
import numpy as np
from get_model.dataset.zarr_dataset import (InferenceDataset,
                                            InferenceReferenceRegionDataset,
                                            ReferenceRegionMotif,
                                            ReferenceRegionMotifConfig)
from modules.caesar.caesar.io.zarr_io import CelltypeDenseZarrIO

random.seed(0)

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
gene_list = np.loadtxt(
    '../genes.txt', dtype=str)

dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=gene_list)
#%%
# %%
cfg = ReferenceRegionMotifConfig(
    data='fetal_k562_peaks_motif.hg38.zarr',
    motif_scaler=1.3)
rrm = ReferenceRegionMotif(cfg)
# %%
rrd = InferenceReferenceRegionDataset(
    rrm, dataset, quantitative_atac=True, sampling_step=100)
# %%
rrd.__len__()
#%%
dataset[0]
# %%
rrd[0]
#%%
def get_distance_map_with_peak_coord(data_dict):
    peak_coord = data_dict['peak_coord']
    
    peak_coord_mean = peak_coord.mean(1)
    # pair-wise distance
    distance = peak_coord_mean[:, None] - peak_coord_mean[None, :]
    return np.absolute(distance)
# %%
distance = get_distance_map_with_peak_coord(rrd[0])
# %%
sns.heatmap(np.log10(distance+1))
# %%
sns.heatmap(np.log10(rrd[0]['hic_matrix']*1000+1))
# %%
# show distance in lower triangle and hic in upper triangle
data = rrd[3]
distance = get_distance_map_with_peak_coord(data)
# distance = np.tril(distance)
hic = data['hic_matrix']
hic = np.log10(hic*1000+1)
# hic = np.triu(hic)
# sns.heatmap(distance + (hic+2)**2, square=True)
# %%
# powerlaw of distance
hic_gamma = 1.024238616787792
hic_scale = 5.9594510043736655
hic_gamma_reference = 0.87
hic_scale_reference = -4.80 + 11.63 * hic_gamma_reference

def get_powerlaw_at_distance(distances, gamma, scale, min_distance=5000):
    assert gamma > 0
    assert scale > 0

    # The powerlaw is computed for distances > 5kb. We don't know what the contact freq looks like at < 5kb.
    # So just assume that everything at < 5kb is equal to 5kb.
    # TO DO: get more accurate powerlaw at < 5kb
    distances = np.clip(distances, min_distance, np.Inf)
    log_dists = np.log(distances + 1)

    powerlaw_contact = np.exp(scale + -1 * gamma * log_dists)
    return powerlaw_contact

sns.histplot(get_powerlaw_at_distance(distance.flatten(), hic_gamma_reference, hic_scale_reference))
# %%
sns.histplot(hic.flatten())
# %%
distance_near = get_powerlaw_at_distance(np.absolute(distance), hic_gamma_reference, hic_scale_reference)
distance_near = distance_near/distance_near.max()
distance_rear = np.log10(np.absolute(distance)+1)
distance_rear = distance_rear/distance_rear.max()
final_distance = (distance_near*0.1 + distance_rear)
final_distance = np.tril(final_distance-final_distance.min())
hic = np.triu(hic)
hic = np.nan_to_num(hic)
hic = hic/hic.max()


sns.heatmap(final_distance + hic, square=True)
# %%
sns.histplot(final_distance.flatten())
# %%
# load yaml config from /home/xf2217/Projects/get_model/get_model/config/model/GETRegionFinetuneExpHiCABC.yaml
from omegaconf import OmegaConf
config = OmegaConf.load('/home/xf2217/Projects/get_model/get_model/config/model/GETRegionFinetuneExpHiCABC.yaml')
import hydra
model = hydra.utils.instantiate(config)['model']
# load GETRegionFinetune_k562_abc/kb3g6ciz/checkpoints/best.ckpt 
# checkpoint = torch.load('/home/xf2217/Projects/get_model/GETRegionFinetune_k562_abc/idhidt4u/checkpoints/best.ckpt')
checkpoint = torch.load('/home/xf2217/Projects/get_model/GETRegionFinetune_k562_abc/2jam37sz/checkpoints/best.ckpt')
from get_model.utils import rename_lit_state_dict
from minlora.model import add_lora_by_name
lora_config = {  # specify which layers to add lora to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=8),
    },
}

add_lora_by_name(model, ['head_exp', 'region_embed', 'encoder'], lora_config)

model.load_state_dict(rename_lit_state_dict(checkpoint['state_dict']))

#%%
from omegaconf import OmegaConf
config = OmegaConf.load('/home/xf2217/Projects/get_model/get_model/config/model/DistanceContactMap.yaml')
import hydra
model = hydra.utils.instantiate(config)['model']
checkpoint = torch.load('/home/xf2217/Projects/get_model/GETRegionFinetune_k562_abc/k6tj3d9k/checkpoints/best.ckpt')
model.load_state_dict(rename_state_dict(checkpoint['state_dict'], {'model.': ''}))
# %%

rrd = ReferenceRegionDataset(rrm, dataset, quantitative_atac=True, sampling_step=100)
# %%
dl = torch.utils.data.DataLoader(rrd, batch_size=1, shuffle=False)
# %%
from tqdm import tqdm
model.to('cuda')
model.eval()
for i, data in tqdm(enumerate(dl)):
    result = model(model.get_input(data)['region_motif'].cuda(), model.get_input(data)['distance_map'].cuda())
    peaks = pd.DataFrame(data['peak_coord'].squeeze().numpy(), columns=['Start', 'End'])
    peaks['Chromosome'] = data['chromosome'][0]
    all_tss_peak = data['all_tss_peak'].squeeze().numpy()
    abc = result[2].squeeze().detach().cpu().numpy()[all_tss_peak]
    # if has 2 dim
    if len(abc.shape) == 2:
        abc = abc.max(0)

    atac = data['region_motif'].squeeze().detach().cpu().numpy()[ :, -1]
    
    peaks['abc'] = abc
    peaks['atac'] = atac
    peaks['gene_name'] = data['gene_name'][0]
    # peaks.to_csv(f"abc/hic_{data['gene_name'][0]}_qatac.csv")

# %%
def plot_diag_heatmap(m1, m2):
    """Plot m1 in lower triangle and m2 in upper triangle"""
    m = np.zeros((m1.shape[0], m1.shape[1]))
    m[np.tril_indices_from(m, -1)] = m1[np.tril_indices_from(m1, -1)]
    m[np.triu_indices_from(m, 1)] = m2[np.triu_indices_from(m2, 1)]
    sns.heatmap(m, square=True, vmax=1)

hic = np.log10(data['hic_matrix'].squeeze().numpy()*1000+1)
atac = data['region_motif'][:,:,-1]
peak_coord = data['peak_coord']
peak_coord_mean = peak_coord[:, :, 0]
# pair-wise distance using torch
    # Add new dimensions to create column and row vectors
peak_coord_mean_col = peak_coord_mean.unsqueeze(2)  # Adds a new axis (column vector)
peak_coord_mean_row = peak_coord_mean.unsqueeze(1)  # Adds a new axis (row vector)

# Compute the pairwise difference
distance = torch.log10((peak_coord_mean_col - peak_coord_mean_row).abs() + 1).squeeze().numpy()
atac = np.sqrt(atac.unsqueeze(1) * atac.unsqueeze(2))
atac = atac.squeeze().numpy()
abc = atac*(1/distance)*10

plot_diag_heatmap(abc, hic)
 # %%

sns.scatterplot(x=hic.flatten()-result[1].squeeze().detach().numpy().flatten(), y=result[1].squeeze().detach().numpy().flatten(), s=1)
# add pearson
from scipy.stats import pearsonr
pearsonr(hic.flatten()-result[1].squeeze().detach().numpy().flatten(), result[1].squeeze().detach().numpy().flatten())
# %%
plot_diag_heatmap(hic, abc*10)
# %%
def plot_bed(bed, query, chrom, start, end, ax=None):
    bed_to_plot = bed.query(
        'Chromosome == @chrom and Start >= @start and End <= @end').copy()
    bed_to_plot = bed_to_plot.query(query)
    bed_to_plot['Start'] = bed_to_plot['Start'] - start
    bed_to_plot['End'] = bed_to_plot['End'] - start
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 5))
    # plot each row in the bed as a rectangle, x range from start to end, y range from 0 to Score,
    # different Name will have different color
    colors = sns.color_palette(
        'mako_r', n_colors=len(bed_to_plot['Name'].unique()))
    colors = {name: color for name, color in zip(
        bed_to_plot['Name'].unique(), colors)}
    for idx, row in bed_to_plot.iterrows():
        ax.fill_between([row['Start'], row['End']], 0,
                        row['Score'], color=colors[row['Name']])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

# %%
df = dataset.datapool.peaks_dict['k562.encode_hg38atac.ENCFF128WZG.max']
df['Name'] = 'bed'
df['Score'] = df['aTPM']

plot_bed(df, 'Count>10', 'chr1', 1000000, 1500000)
# %%
from caesar.io.zarr_io import CelltypeDenseZarrIO
cdz = CelltypeDenseZarrIO('/home/xf2217/Projects/get_data/encode_hg38atac_dense.zarr')
# %%
df_specific = cdz.get_peaks('k562.encode_hg38atac.ENCFF128WZG.max', name='peaks_q0.05_tissue_open')
df_specific['Name'] = 'specific'
df_specific['Score'] = df_specific['aTPM']
plot_bed(df_specific, 'Count>10', 'chr1', 1000000, 1500000)

#%%
fig, ax = plt.subplots(2, 1, figsize=(20, 6))


plot_bed(df, 'Count>10', 'chr1', 1000000, 1100000, ax=ax[0])
plot_bed(df_specific, 'Count>10', 'chr1', 1000000, 1100000, ax=ax[1])

#%%
def plot_panel(chrom, start, end, bed1, bed2, query, ids=None):
    if ids == None:
        ids = cdz.ids
    fig, axes = plt.subplots(len(ids)+2, 1, figsize=(
        20, 6), sharex=True, sharey=False)
    for i, cell_type in enumerate(ids):
        z = cdz.get_track(cell_type, chrom, start, end)
        # conv 50
        z = np.convolve(z, np.ones(50)/50, mode='same')
        # plot bed as a overlay
        axes[i].plot(z)
    plot_bed(bed1, query,
             chrom, start, end, ax=axes[-2])
    plot_bed(bed2, query,
             chrom, start, end, ax=axes[-1])
    return fig, axes

plot_panel('chr2', 890000, 3100000, df, df_specific, 'Count>10', ids=None)



# %%
cdz.da['k562.encode_hg38atac.ENCFF257HEE.max']
# %%
