# %%
from caesar.io.zarr_io import CelltypeDenseZarrIO

# %%
encode_data = "/pmglocal/alb2281/get/get_data/encode_hg38atac_dense.zarr"
# %%

cdz = CelltypeDenseZarrIO('/pmglocal/alb2281/get/get_data/encode_hg38atac_dense.zarr', 'a')


# %%
cdz.ids
# %%

rna_peaks = cdz.get_peaks('k562.encode_hg38atac.ENCFF128WZG.max', name='peaks_q0.05_fetal_joint_tissue_open_exp')
# %%

cage_peaks = cdz.get_peaks('k562.encode_hg38atac.ENCFF128WZG_cage.max', name='peaks_q0.05_fetal_joint_tissue_open_exp')
#%%
cage_values = cage_peaks[['Expression_positive', 'Expression_negative']].values

# %%
merged_peaks = rna_peaks.merge(cage_peaks, on=["Chromosome", "Start", "End"], suffixes=('_rna', '_cage'))
# %%
import seaborn as sns
# %%
sns.scatterplot(data=merged_peaks, x='Expression_positive_rna', y='Expression_positive_cage', s=1)
# %%
sns.scatterplot(data=merged_peaks, x='Expression_negative_rna', y='Expression_negative_cage', s=1)
# %%
# compute pearson correlation
from scipy.stats import pearsonr
pearsonr(merged_peaks['Expression_positive_rna'], merged_peaks['Expression_positive_cage'])
# %%
pearsonr(merged_peaks['Expression_negative_rna'], merged_peaks['Expression_negative_cage'])
# %%


## 

cage_peaks_subset = cage_peaks[(cage_peaks['Chromosome'] == "chr10") | (cage_peaks['Chromosome'] == "chr11")]

# %%
cage_peaks_subset.to_csv("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/cage_peaks_leaveout_k562.csv", index=False)
# %%
