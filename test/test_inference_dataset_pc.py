# %%
import matplotlib.pyplot as plt
import random
import pyliftover

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from caesar.io.gencode import Gencode
from caesar.io.zarr_io import DenseZarrIO
from pyranges import PyRanges as pr
from tqdm import tqdm

from get_model.dataset.zarr_dataset import InferenceDataset
from get_model.inference_engine import InferenceEngine

random.seed(0)

# %%
# Configuration for Gencode
gencode_config = {
    "assembly": "hg38",
    "version": 40,
    "gtf_dir": "/home/xf2217/Projects/caesar/data/"
}
model_checkpoint = "/pmglocal/xf2217/Expression_Finetune_monocyte.Chr4&14.conv50.learnable_motif_prior.chrombpnet.shift10.R100L1000.augmented.20240307/checkpoint-best.pth"
# Configuration for the dataset
dataset_config = {
    "zarr_dirs": ["/home/xf2217/Projects/get_data/encode_hg38atac_dense.zarr"],
    "genome_seq_zarr": {'hg38': "/home/xf2217/Projects/get_data/hg38.zarr"},
    "genome_motif_zarr": "/home/xf2217/Projects/get_data/hg38_motif_result.zarr",
    "insulation_paths": [
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather",
        "/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather"
    ],
    "peak_name": "peaks_p0.01_tissue_open_exp",
    "additional_peak_columns": ["Expression_positive", "Expression_negative", "aTPM", "TSS"],
    "n_peaks_upper_bound": 100,
    "center_expand_target": 0,
    "padding": 0,
}
# %%
hg38 = DenseZarrIO('/home/xf2217/Projects/get_data/hg38.zarr')
gencode = Gencode(**gencode_config)
dataset = InferenceDataset(
    assembly='hg38', gencode_obj=gencode, return_all_tss=True, **dataset_config)
# %%
data = dataset.__getitem__(10535)
# %%
data
# %%
track = data['sample']['sample_track'].toarray().flatten()
# conv 50
track = np.convolve(track, np.ones(50)/50, mode='same')
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(track)
# %%
cdz = dataset.datapool.zarr_dict['encode_hg38atac_dense.zarr']
# %%
original_track = cdz.get_track(cdz.ids[0], data['sample']['metadata']['chr_name'],
                               data['sample']['metadata']['start'], data['sample']['metadata']['end'])
original_track = np.convolve(original_track, np.ones(50)/50, mode='same')
# %%
cdz.get_peaks(cdz.ids[0], 'peaks_p0.05_tissue_open_exp')

# %%
reconstructed_track = np.zeros_like(original_track)
for i, peak in enumerate(data['sample']['celltype_peaks']):
    start, end = peak
    track_start = i * 500
    track_end = track_start + end - start
    print(track_start, track_end)
    reconstructed_track[start:end] = data['sample']['sample_track'].toarray().flatten()[
        track_start:track_end]

# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 2))
reconstructed_track = np.convolve(
    reconstructed_track, np.ones(50)/50, mode='same')
ax[0].plot(reconstructed_track)
ax[1].plot(original_track)
# %%
# plot gencode
data
# %%


class SampleHandler(dict):
    def __init__(self, *args, **kwargs):
        super(SampleHandler, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        return self[key]

    def plot_sample_track(self, conv=50, figsize=(10, 2)):
        track = self.sample_track.toarray().flatten()
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(np.convolve(track, np.ones(conv)/conv, mode='same'))
        return track

    def plot_reconstruct_track(self, conv=50, figsize=(10, 2)):
        reconstructed_track = np.zeros(
            (self.metadata['end'] - self.metadata['start']))
        peak_boundary = self.celltype_peaks[:, 1] - self.celltype_peaks[:, 0]
        peak_boundary = np.cumsum([0] + list(peak_boundary))[0:-1]
        peak_length = np.array(
            [end - start for start, end in self.celltype_peaks])
        for i, peak in enumerate(self.celltype_peaks):
            start, end = peak
            track_start = peak_boundary[i]
            track_end = track_start + peak_length[i]
            print(track_start, track_end)
            reconstructed_track[start:end] = self.sample_track.toarray().flatten()[
                track_start:track_end]
        fig, ax = plt.subplots(figsize=figsize)
        reconstructed_track = np.convolve(
            reconstructed_track, np.ones(conv)/conv, mode='same')
        ax.plot(reconstructed_track)
        return reconstructed_track


sample = SampleHandler(data)
# %%
sample.plot_sample_track()
# %%
sample.plot_reconstruct_track()
# %%
