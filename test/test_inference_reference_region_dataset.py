# %%
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
    "zarr_dirs": ["/home/xf2217/Projects/get_data/joung_tfatlas_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/home/xf2217/Projects/get_data/hg38.zarr"},
    "genome_motif_zarr": "/home/xf2217/Projects/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "fetal_tfatlas_peaks_open_exp",
    "leave_out_chromosomes": "",
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
rrd.__len__()

# %%
rrd[0]

# %%
rrd.inference_dataset.datapool.peaks_dict['0.joung_tfatlas.shareseq.8192'].iloc[485395+450]
# %%
rrd.inference_dataset[0]
# %%
rrd.data_dict['0.joung_tfatlas.shareseq.8192'][1].reset_index().query(
    'level_0!=index_input')

# %%
rrd.inference_dataset.datapool.peaks_dict['0.joung_tfatlas.shareseq.8192']
# %%
# peak in the dataset and the peak in the data_dict are the same
rrd.data_dict['0.joung_tfatlas.shareseq.8192'][1]['aTPM'] == rrd.inference_dataset.datapool.peaks_dict['0.joung_tfatlas.shareseq.8192']['aTPM']
# %%
rrd.data_dict['0.joung_tfatlas.shareseq.8192'][0][485395+450]
# %%
rrm.data[485395+450]
# %%
import seaborn as sns
sns.scatterplot(x=rrd.data_dict['0.joung_tfatlas.shareseq.8192'][0][485395+450])
# %%
# %%
