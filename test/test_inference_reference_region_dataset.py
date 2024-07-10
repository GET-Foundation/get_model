# %%
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
    "zarr_dirs": ["/home/xf2217/Projects/get_data/htan_gbm_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/home/xf2217/Projects/get_data/hg38.zarr"},
    "genome_motif_zarr": "/home/xf2217/Projects/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "fetal_gbm_peaks_open_exp",
    "leave_out_chromosomes": "",
    "use_insulation": True,
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 200,
    "keep_celltypes": "Tumor.htan_gbm.C3N-02783_CPT0205890014_snATAC_GBM_Tumor.1024",
    "center_expand_target": 0,
    "random_shift_peak": 0,
    "peak_count_filter": 2,
    'mask_ratio': 0,
    "padding": 0,
    "hic_path": None #"/home/xf2217/Projects/get_data/GSE206131_K562_DMSO.mapq_30.mcool"
}
# %%
hg38 = DenseZarrIO('/home/xf2217/Projects/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)
# %%
gene_list = np.loadtxt(
    '/home/xf2217/Projects/get_revision/TFAtlas_fetal_compare/diff_genes.txt', dtype=str)[0:10]
gene_list = ['EGFR', 'TP53', 'ABCC9']
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
rrd.__len__()

# %%
sum(rrd.zarr_dataset[0]['additional_peak_features']
    [:, 0:2] == rrd[0]['exp_label'])
# %%
rrd.data_dict['0.joung_tfatlas.L10M'][1].loc[rrd.zarr_dataset[0]
                                             ['metadata']['original_peak_start']]

datapool_peak = rrd.zarr_dataset.datapool.peaks_dict['0.joung_tfatlas.L10M']

rrd_peak = rrd.data_dict['0.joung_tfatlas.L10M'][1]

tss_coord = rrd.zarr_dataset[0]['metadata']['tss_coord']

peak_start = rrd.zarr_dataset[0]['metadata']['original_peak_start']

tss_peak = rrd.zarr_dataset[0]['metadata']['tss_peak']

tss_peak_df = datapool_peak.iloc[peak_start+tss_peak]
# %%
print(tss_coord > tss_peak_df.Start and tss_coord < tss_peak_df.End)
# %%
def extract_peak_df(batch):
    peak_coord = batch['celltype_peaks'] + batch['metadata']['start'] 
    chr_name = batch['metadata']['chr_name']
    df = pd.DataFrame(peak_coord, columns=['Start', 'End'])
    df['Chromosome'] = chr_name
    return df[['Chromosome', 'Start', 'End']]

def get_insulation_overlap(batch, insulation):
    from pyranges import PyRanges as pr
    peak_df = extract_peak_df(batch)
    insulation = insulation[insulation['Chromosome'] == peak_df['Chromosome'].values[0]]
    overlap = pr(peak_df.iloc[batch['metadata']['tss_peak']]).join(pr(insulation), suffix='_insulation').df
    final_insulation = overlap.sort_values('mean_num_celltype').iloc[-1][['Chromosome', 'Start_insulation', 'End_insulation']].rename({'Start_insulation': 'Start', 'End_insulation': 'End'})
    subset_peak_df = peak_df.loc[(peak_df.Start>final_insulation.Start) & (peak_df.End<final_insulation.End)]
    new_peak_start_idx = subset_peak_df.index.min()
    new_peak_end_idx = subset_peak_df.index.max()
    new_tss_peak = batch['metadata']['tss_peak'] - new_peak_start_idx
    return new_peak_start_idx, new_peak_end_idx, new_tss_peak


# %%
get_insulation_overlap(dataset[0], dataset.datapool.insulation)
# %%
