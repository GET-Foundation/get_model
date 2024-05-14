# %%
from http.client import OK
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
    "gtf_dir": "/pmglocal/xf2217/GET_STARTED//caesar/data/"
}
# Configuration for the dataset
dataset_config = {
    "zarr_dirs": ["/pmglocal/xf2217/get_data/encode_hg38atac_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/pmglocal/xf2217/get_data/hg38.zarr"},
    "genome_motif_zarr": "/pmglocal/xf2217/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/pmglocal/xf2217/GET_STARTED//get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/pmglocal/xf2217/GET_STARTED//get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "peaks_q0.05_fetal_joint_tissue_open_exp",
    "keep_celltypes": "k562.encode_hg38atac.ENCFF128WZG.max",
    "leave_out_chromosomes": "",
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 900,
    "center_expand_target": 0,
    "random_shift_peak": 0,
    'mask_ratio': 0,
    "padding": 0,
}
# %%
hg38 = DenseZarrIO('/pmglocal/xf2217/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)
# %%
gene_list = np.loadtxt(
    '/burg/pmg/users/xf2217/CRISPR_comparison/genes.txt', dtype=str)
dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=gene_list)
# %%
cfg = ReferenceRegionMotifConfig(
    root='/pmglocal/xf2217/get_data/',
    data='fetal_k562_peaks_motif.hg38.zarr',
    refdata='fetal_union_peak_motif_v1.hg38.zarr',
    count_filter=10,
    motif_scaler=1.3)
rrm = ReferenceRegionMotif(cfg)
# %%
rrd = InferenceReferenceRegionDataset(
    rrm, dataset, quantitative_atac=True, sampling_step=450)
# %%
rrd.__len__()

# %%
rrd.data_dict['k562.encode_hg38atac.ENCFF128WZG.max'][1].query('Chromosome=="chr11" & Start>5504223 & End<5507000')

# %%
InferenceReferenceRegionDataset
- InferenceDataset 
    - peak_orig -> ( aTPM filter ) -> open peaks
    - gene_list
- ReferenceRegionMotif
    - K562+Fetal
        - region_motif matrix
        - peak_orig   
    - Fetal reference
        - region_motif matrix
        - peak   