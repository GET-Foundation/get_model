# %%
import pandas as pd 
# %%
names = ["chr", "start", "end", "split"]
basenji_human_splits = "/burg/pmg/users/alb2281/enformer/basenji_splits/enformer_human_sequences.bed"
# %%
basenji_human_splits_df = pd.read_csv(basenji_human_splits, sep="\t", names=names)
# %%
from caesar.io.zarr_io import CelltypeDenseZarrIO

# %%
encode_data = "/pmglocal/alb2281/get/get_data/encode_hg38atac_dense.zarr"
# %%

cdz = CelltypeDenseZarrIO('/pmglocal/alb2281/get/get_data/encode_hg38atac_dense.zarr', 'a')


# %%
cdz.ids
# %%
# %%

cage_peaks = cdz.get_peaks('k562.encode_hg38atac.ENCFF128WZG_cage.max', name='peaks_q0.05_fetal_joint_tissue_open_exp')
# %%

