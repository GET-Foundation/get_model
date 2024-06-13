# %%
import torch
from omegaconf import OmegaConf
from get_model.utils import rename_state_dict
config = OmegaConf.load('/home/xf2217/Projects/get_model/get_model/config/model/DistanceContactMap.yaml')
import hydra
model = hydra.utils.instantiate(config)['model']
checkpoint = torch.load('/home/xf2217/Projects/get_model/GETRegionFinetune_k562_abc/k6tj3d9k/checkpoints/best.ckpt')
model.load_state_dict(rename_state_dict(checkpoint['state_dict'], {'model.': ''}))
# %%
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
    "n_peaks_upper_bound": 200,
    "keep_celltypes": "k562.encode_hg38atac.ENCFF128WZG.max",
    "center_expand_target": 0,
    "random_shift_peak": 0,
    "peak_count_filter": 10,
    'mask_ratio': 0,
    "padding": 0,
    "hic_path": None
}
# %%
hg38 = DenseZarrIO('/home/xf2217/Projects/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)
# %%
gene_list = np.loadtxt(
    '../genes.txt', dtype=str)

dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=gene_list)
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
# %%

rrd = ReferenceRegionDataset(rrm, dataset, quantitative_atac=True, sampling_step=100)
# %%
dl = torch.utils.data.DataLoader(rrd, batch_size=2, shuffle=False)
# %%
for i, data in enumerate(dl):
    print(i)
    # result = model(model.get_input(data)['distance_map']).squeeze().detach().numpy()

    if i == 0:
        break

    # peaks = pd.DataFrame(data['peak_coord'].squeeze().numpy(), columns=['Start', 'End'])
    # peaks['Chromosome'] = data['chromosome'][0]
    # all_tss_peak = data['all_tss_peak'].squeeze().numpy()
    # abc = result[2].squeeze().detach().numpy()[all_tss_peak]
    # # if has 2 dim
    # if len(abc.shape) == 2:
    #     abc = abc.mean(0)

    # atac = data['region_motif'].squeeze().numpy()[ :, -1]
    
    # peaks['abc'] = abc
    # peaks['atac'] = atac
    # peaks['gene_name'] = data['gene_name'][0]
    # peaks.to_csv(f"abc/{data['gene_name'][0]}.csv")

# %%
def plot_diag_heatmap(m1, m2):
    """Plot m1 in lower triangle and m2 in upper triangle"""
    m = np.zeros((m1.shape[0], m1.shape[1]))
    m[np.tril_indices_from(m, -1)] = m1[np.tril_indices_from(m1, -1)]
    m[np.triu_indices_from(m, 1)] = m2[np.triu_indices_from(m2, 1)]
    sns.heatmap(m, square=True, vmax=1, vmin=0)

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
abc = atac*result

plot_diag_heatmap(abc, atac*hic-abc)
# %%

sns.scatterplot(x=abc.flatten(), y=(atac*hic).flatten(), s=1)
# add pearson
from scipy.stats import pearsonr
pearsonr(hic.flatten()-result[1].squeeze().detach().numpy().flatten(), result[1].squeeze().detach().numpy().flatten())
# %%
plot_diag_heatmap(hic, abc*10)
# %%
