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

# %%
cdz = CelltypeDenseZarrIO('/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr')
# %%
cdz = cdz.subset_celltypes_with_data_name()
#%%
cdz.ids



#%%
pretrain = PretrainDataset(['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr',
                            ],
                           '/pmglocal/xf2217/get_data/hg38.zarr', 
                           '/pmglocal/xf2217/get_data/hg38_motif_result.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue_open_exp', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=1000, n_peaks_lower_bound=5, n_peaks_upper_bound=200)
pretrain.__len__()
# %%
pretrain.datapool.peaks_dict
# %%
