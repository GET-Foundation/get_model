# %%
import pandas as pd
import seaborn as sns
import torch.utils
from get_model.run_ref_region import *
import random

from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO
import numpy as np
from get_model.dataset.zarr_dataset import (InferenceDataset,
                                            InferenceReferenceRegionDataset,
                                            PerturbationInferenceReferenceRegionDataset,
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
    "zarr_dirs": ["/home/xf2217/Projects/get_data/joung_tfatlas_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/home/xf2217/Projects/get_data/hg38.zarr"},
    "genome_motif_zarr": "/home/xf2217/Projects/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "fetal_tfatlas_peaks_tissue_open_exp",
    "keep_celltypes": "0.joung_tfatlas.L10M",
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 900,
    "center_expand_target": 0,
    'mask_ratio': 0,
    "padding": 0,
}
# %%
hg38 = DenseZarrIO('/home/xf2217/Projects/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)

gene_list = np.loadtxt(
    '/home/xf2217/Projects/get_revision/TFAtlas_fetal_compare/diff_genes.txt', dtype=str)
dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=gene_list)
# %%
cfg = ReferenceRegionMotifConfig()
rrm = ReferenceRegionMotif(cfg)
# %%
rrd = InferenceReferenceRegionDataset(
    rrm, dataset, quantitative_atac=True, sampling_step=450)
# %%

# chr1: 29200-29505
inactivated_peaks = pd.DataFrame({
    'Chromosome': ['chr4'],
    'Start': [1309046+1980],
    'End': [1309046+1150980],
    'Strand': ['+'],
})
# %%
pirrd = PerturbationInferenceReferenceRegionDataset(
    inference_dataset=rrd, perturbations=inactivated_peaks, mode='peak_inactivation')

# %%
item = pirrd[2]

# %%
# Verify that the specified peaks are inactivated (set to 0) in the MUT sample
mut_region_motif = item['MUT']['region_motif']
wt_region_motif = item['WT']['region_motif']
print("Region Motif WT:\n", wt_region_motif)
print("Region Motif MUT:\n", mut_region_motif)
# %%
sns.heatmap(mut_region_motif-wt_region_motif)
# %%
