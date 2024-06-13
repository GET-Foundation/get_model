# %%
from tqdm import tqdm
from get_model.dataset.zarr_dataset import (InferenceDataset, InferenceEverythingDataset,
                                            ReferenceRegionMotif,
                                            ReferenceRegionMotifConfig)
import torch.utils
from get_model.run_ref_region import *
import random

from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO
import numpy as np
np.bool = np.bool_

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
    "genome_motif_zarr": None,
    "insulation_paths": [
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "peaks_q0.05_fetal_joint_tissue_open_exp",
    "leave_out_chromosomes": None,
    "use_insulation": True,
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 200,
    "keep_celltypes": "k562.encode_hg38atac.ENCFF257HEE.max",
    "center_expand_target": 0,
    "random_shift_peak": 0,
    "peak_count_filter": 0,
    'mask_ratio': 0,
    "padding": 0,
    "hic_path": "/home/xf2217/Projects/encode_hg38atac/raw/ENCFF621AIY.hic"
}
# %%
hg38 = DenseZarrIO('/home/xf2217/Projects/get_data/hg38.zarr', dtype='int8')
gencode = Gencode(**gencode_config)
# %%
gene_list = np.loadtxt(
    '../genes.txt', dtype=str)

dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=gene_list)
# %%
cfg = ReferenceRegionMotifConfig(
    data='k562.ENCFF257HEE.encode_hg38atac.peak_motif.zarr',
    motif_scaler=1.3)
rrm = ReferenceRegionMotif(cfg)
# %%
everything = InferenceEverythingDataset(
    rrm, dataset, quantitative_atac=True, sampling_step=450)
#%%
from get_model.dataset.collate import everything_collate

dl = torch.utils.data.DataLoader(everything, batch_size=2, collate_fn=everything_collate, num_workers=16, shuffle=False)

# %%
for i, batch in tqdm(enumerate(dl)):
    if i > 1000:
        break

# %%
