#%%
import hicstraw
import numpy as np  
hic = hicstraw.HiCFile("../data/4DNFI2TK7L2F.hic")
#%%
def get_hic_from_idx(hic, csv, start, end, resolution=10000, method='observed'):
    csv_region = csv.iloc[start:end]
    chrom = csv_region.iloc[0].Chromosome.replace("chr", "")
    if chrom != csv_region.iloc[-1].Chromosome.replace("chr", ""):
        return None
    start = csv_region.iloc[0].Start // resolution
    end = csv_region.iloc[-1].End // resolution + 1
    if (end-start) * resolution > 4000000:
        return None
    hic_idx = np.array([row.Start // resolution - start + 1 for _, row in csv_region.iterrows()])
    mzd = hic.getMatrixZoomData(chrom, chrom, method, "KR", "BP", resolution)
    numpy_matrix = mzd.getRecordsAsMatrix(start * resolution, end * resolution, start * resolution, end * resolution)
    dst = np.log2(numpy_matrix[hic_idx,:][:, hic_idx]+1)
    return dst

# %%
import logging
from matplotlib import pyplot as plt

import numpy as np
import seaborn as sns
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack, CelltypeDenseZarrIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
#%% 
pretrain = PretrainDataset(zarr_dirs=['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr',
                            ],
                           genome_seq_zarr='/pmglocal/xf2217/get_data/hg38.zarr', 
                           genome_motif_zarr='/pmglocal/xf2217/get_data/hg38_motif_result.zarr', insulation_paths=[
                           '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], , peak_name='peaks_q0.01_tissue_open_exp', preload_count=100, n_packs=1,
                           max_peak_length=5000, center_expand_target=500, n_peaks_lower_bound=10, n_peaks_upper_bound=100, use_insulation=True, leave_out_celltypes='Astrocyte', leave_out_chromosomes='chr1', is_train=False, dataset_size=65536, additional_peak_columns=None)
# %%
df = pretrain.datapool.peaks_dict[list(pretrain.datapool.peaks_dict.keys())[0]]
# %%
from tqdm import tqdm
for i in tqdm(range(100)):
    arr = get_hic_from_idx(hic, df, 0, 100)
# %%
arr = get_hic_from_idx(hic, df, 5000, 5100, 5000, 'observed')
import seaborn as sns
# plot Count in df as barplot above the heatmap
fig, ax = plt.subplots(2, 1, figsize=(3, 4), gridspec_kw={'height_ratios': [1, 4]})
aTPM = df.iloc[5000:5100].aTPM.values
ax[0].bar(np.arange(100), aTPM)
ax[0].set_xlim(0, 100)
sns.heatmap(arr, ax=ax[1], cbar=False)
# %%
