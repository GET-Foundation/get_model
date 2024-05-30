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
    "gtf_dir": "/pmglocal/alb2281/get/get_data"
}
# Configuration for the dataset
dataset_config = {
    "zarr_dirs": ["/pmglocal/alb2281/get/get_data/htan_gbm_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/pmglocal/alb2281/get/get_data/hg38.zarr"},
    "genome_motif_zarr": "/pmglocal/alb2281/get/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/pmglocal/alb2281/repos/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/pmglocal/alb2281/repos/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "fetal_gbm_peaks_open_exp",
    "leave_out_chromosomes": "",
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 900,
    "keep_celltypes": "Tumor.htan_gbm.C3N-01814_CPT0167860015_snATAC_GBM_Tumor.16384",
    "center_expand_target": 0,
    "random_shift_peak": 0,
    "peak_count_filter": 3,
    'mask_ratio': 0,
    "padding": 0,
}
# %%
hg38 = DenseZarrIO('/pmglocal/alb2281/get/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)
# %%
# gene_list = np.loadtxt(
#     '/home/xf2217/Projects/get_revision/TFAtlas_fetal_compare/diff_genes.txt', dtype=str)[0:10]
gene_list = ['AARSD1', 'MYC', 'RET']
dataset = InferenceDataset(
    assembly='hg38', gencode_obj={'hg38': gencode}, **dataset_config, gene_list=gene_list)
# %%
cfg = ReferenceRegionMotifConfig(
    motif_scaler=1.3)
rrm = ReferenceRegionMotif(cfg)
# %%
rrd = InferenceReferenceRegionDataset(
    rrm, dataset, quantitative_atac=True, sampling_step=450)
# %%
rrd[0]

# %%
sum(rrd.zarr_dataset[0]['additional_peak_features']
    [:, 0:2] == rrd[0]['exp_label'])
# %%
rrd.data_dict['Tumor.htan_gbm.C3N-01814_CPT0167860015_snATAC_GBM_Tumor.16384'][1].loc[rrd.zarr_dataset[0]
                                             ['metadata']['original_peak_start']+450]

datapool_peak = rrd.zarr_dataset.datapool.peaks_dict['Tumor.htan_gbm.C3N-01814_CPT0167860015_snATAC_GBM_Tumor.16384']

rrd_peak = rrd.data_dict['Tumor.htan_gbm.C3N-01814_CPT0167860015_snATAC_GBM_Tumor.16384'][1]

tss_coord = rrd.zarr_dataset[0]['metadata']['tss_coord']

peak_start = rrd.zarr_dataset[0]['metadata']['original_peak_start']

tss_peak = rrd.zarr_dataset[0]['metadata']['tss_peak']

tss_peak_df = datapool_peak.iloc[peak_start+tss_peak]
# %%
print(tss_coord > tss_peak_df.Start and tss_coord < tss_peak_df.End)
# %%
rrd_peak.query('index_input==@peak_start+@tss_peak')
# %%
tss_peak_df.Start==rrd_peak.query('index_input==@peak_start+@tss_peak').Start
# %%
rrd[1]
# %%
