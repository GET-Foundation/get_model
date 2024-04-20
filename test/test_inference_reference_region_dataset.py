# %%
import random

from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO

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
    "leave_out_chromosomes": ['chr10', 'chr11'],
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 900,
    "center_expand_target": 0,
    'mask_ratio': 0,
    "padding": 0,
}
# %%
hg38 = DenseZarrIO('/home/xf2217/Projects/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)
dataset = InferenceDataset(
    assembly='hg38', gencode_obj=gencode, **dataset_config, gene_list=None)
# %%
data = dataset.__getitem__(0)
# %%
cfg = ReferenceRegionMotifConfig()
rrm = ReferenceRegionMotif(cfg)
# %%
rrd = InferenceReferenceRegionDataset(
    rrm, dataset, quantitative_atac=True, sampling_step=450)
# %%
d = rrd.__getitem__(1000)

# %%
