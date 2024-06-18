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
merged_df = rna_peaks.merge(cage_peaks, on=['Chromosome', 'Start', 'End'], suffixes=('_rna', '_cage'))

# %%
sns.scatterplot(data=merged_df, x='Expression_positive_cage', y='Expression_positive_rna', s=1)

# Compute pearson between cage and rna expression
merged_df[['Expression_positive_cage', 'Expression_positive_rna']].corr()
merged_df[['Expression_negative_cage', 'Expression_negative_rna']].corr()
#%%
chr_order = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
# reorder cage_peaks
cage_peaks = cage_peaks.set_index('Chromosome').loc[chr_order].reset_index()
rna_peaks = rna_peaks.set_index('Chromosome').loc[chr_order].reset_index()
#%%
cage_peaks['Expression_positive'] = cage_values[:,0]
cage_peaks['Expression_negative'] = cage_values[:,1]
# %%
cage_peaks.plot('Expression_positive', 'aTPM', kind='scatter', s=0.1)
# %%
cage_peaks.query('Chromosome=="chr1"').plot('Expression_positive', 'aTPM', kind='scatter', s=0.1)
# %%
merged_df = rna_peaks.copy()
merged_df['cage_exp'] = cage_peaks['Expression_positive'] + cage_peaks['Expression_negative']
merged_df['rna_exp'] = rna_peaks['Expression_positive'] + rna_peaks['Expression_negative']
merged_df.query('rna_exp>0').plot('cage_exp', 'rna_exp', kind='scatter', s=0.1)
# %%
# pearson
merged_df.query('cage_exp>0')[['rna_exp', 'cage_exp']].corr()
# %%
cdz.write_peaks('k562.encode_hg38atac.ENCFF128WZG_cage.max', cage_peaks, 'peaks_q0.05_fetal_joint_tissue_open_exp', ['Start', 'End', 'Expression_positive', 'Expression_negative', 'aTPM', 'Count', 'TSS'], overwrite=True)
# %%
