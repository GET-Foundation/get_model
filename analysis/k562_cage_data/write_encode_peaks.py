# %%
import logging

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack, CelltypeDenseZarrIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# %% 
cdz = CelltypeDenseZarrIO('/pmglocal/alb2281/get/get_data/encode_hg38atac_dense.zarr', 'a')
k562_id = "k562.encode_hg38atac.ENCFF128WZG.max"
peaks = cdz.get_peaks(k562_id, name='peaks_q0.05_fetal_joint_tissue_open_exp')

# %%
peaks = peaks[["Chromosome", "Start", "End"]]

# %%
peaks["Name"] = peaks["Chromosome"] + ":" + peaks["Start"].astype(str) + "-" + peaks["End"].astype(str)

# %%
peaks.to_csv("/pmglocal/alb2281/repos/get_model/analysis/k562_cage_data/k562_encode_peaks.bed", sep="\t", header=False, index=False)

# %%
cage_plus = "/pmglocal/alb2281/repos/get_model/analysis/k562_cage_data/k562.cage_plus.tsv"
cage_minus = "/pmglocal/alb2281/repos/get_model/analysis/k562_cage_data/k562.cage_minus.tsv"
# %%
import pandas as pd

cols = ["name", "size", "covered", "sum", "mean0", "mean"]

cage_plus = pd.read_csv(cage_plus, sep="\t", names=cols)
cage_minus = pd.read_csv(cage_minus, sep="\t", names=cols)
# %%

import numpy as np

cage_plus["tpm"] = np.log10(cage_plus["sum"]/cage_plus["sum"].sum() * 1e6 + 1)
# %%
cage_minus["tpm"] = np.log10(cage_minus["sum"]/cage_minus["sum"].sum() * 1e6 + 1)
# %%

cdz = CelltypeDenseZarrIO('/pmglocal/alb2281/get/get_data/encode_hg38atac_dense.zarr', 'a')

# %%

k562_id = "k562.encode_hg38atac.ENCFF128WZG_cage.max"
peaks = cdz.get_peaks(k562_id, name='peaks_q0.05_fetal_joint_tissue_open_exp')



# %%
peaks["Expression_positive"] = cage_plus["tpm"].values
peaks["Expression_negative"] = cage_minus["tpm"].values

# %%
cdz.write_peaks(
    k562_id, 
    peaks,
    'peaks_q0.05_fetal_joint_tissue_open_exp',
    ['Start', 'End', 'Expression_positive', 'Expression_negative', 'aTPM', 'Count', 'TSS'],
    overwrite=True
)
# %%
