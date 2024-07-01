# %%

from caesar.io.zarr_io import CelltypeDenseZarrIO
import pandas as pd

# %%
enformer_targets = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/targets_human.txt"

# %%

cdz = CelltypeDenseZarrIO("/pmglocal/alb2281/get/get_data/encode_hg38atac_dense.zarr")
# %%

cdz.ids
# %%

peaks = cdz.get_peaks("k562.encode_hg38atac.ENCFF257HEE.max", "peaks_q0.05_fetal_joint_tissue_open_exp")
# %%

peaks_chr_10 = peaks[peaks["Chromosome"] == "chr10"]
# %%

peaks_chr_10.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/encode_ENCFF257HEE_dnase_k562.csv")
# %%

targets_df = pd.read_csv(enformer_targets, sep="\t")
# %%

targets_k562_df = targets_df[targets_df["description"] == "DNASE:K562"]

# %%
indices = targets_k562_df["index"].values
# %%
