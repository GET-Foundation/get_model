#%%
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import snapatac2 as snap
from pyranges import PyRanges as pr

from gcell._settings import get_setting

annotation_dir = get_setting('annotation_dir')
print("gcell currently using annotation directory:", annotation_dir)

# %% [markdown]
"""
# Data Preparation Tutorial from SnapATAC2
This tutorial demonstrates how to prepare single-cell multiome (RNA + ATAC) data for GET model training.
We'll use the PBMC10k dataset from 10x Genomics as an example.

## Overview:
1. Process RNA data - normalize, filter for highly variable genes
2. Process ATAC data - compute spectral embedding
3. Filter for abundant cell types
4. Generate peak accessibility and gene expression files for each cell type
"""

#%% [markdown]
"""
## 1. RNA Data Processing
- Load RNA data from PBMC10k dataset
- Select top 3000 highly variable genes
- Normalize and log transform the data
- Compute spectral embedding and UMAP visualization
"""
# %%
# read rna data
if not Path('rna.h5ad').exists():
    rna = snap.read(snap.datasets.pbmc10k_multiome(modality='RNA'), backed=None)
    sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=3000)
    rna_filtered = rna[:, rna.var.highly_variable]
    sc.pp.normalize_total(rna_filtered, target_sum=1e4)
    sc.pp.log1p(rna_filtered)
    snap.tl.spectral(rna_filtered, features=None)
    snap.tl.umap(rna_filtered)
else:
    rna_filtered = sc.read('rna.h5ad')
#%%
# read atac data
if not Path('atac.h5ad').exists():
    atac = snap.read(snap.datasets.pbmc10k_multiome(modality='ATAC'), backed=None)
    snap.tl.spectral(atac, features=None)
    snap.tl.umap(atac)
    assert (rna_filtered.obs_names == atac.obs_names).all()
    embedding = snap.tl.multi_spectral([rna_filtered, atac], features=None)[1]
    atac.obsm['X_joint'] = embedding
    snap.tl.umap(atac, use_rep='X_joint')
    atac.write('atac.h5ad')
else:
    atac = sc.read('atac.h5ad')

#%%
rna_original = snap.read(snap.datasets.pbmc10k_multiome(modality='RNA'), backed=None)
# %% [markdown]
"""
## 3. Cell Type Selection
We filter for cell types with >1000 cells to ensure robust statistics for modeling.
The selected cell types will be used for training GET models.
"""

# %%
cell_number = atac.obs.groupby('cell_type', observed=False).size().to_dict()
print("The following cell types have more than 1000 cells, adding them to celltype_for_modeling")
celltype_for_modeling = []
for cell_type in cell_number:
    if cell_number[cell_type] > 1000:
        celltype_for_modeling.append(cell_type)
        libsize = int(atac.X[atac.obs.cell_type == cell_type].sum())
        print(f"{cell_type} number of cells: {cell_number[cell_type]}, library size: {libsize}")
# %%
def get_peak_from_snapatac(atac: snap.AnnData):
    """
    Get the peak names from the snapatac object.

    Args:
        atac: snapatac2 processed AnnData object

    Returns:
        peak_names: pandas DatasFrame with the peak names
    """
    peak_names = pd.DataFrame(atac.var.index.str.split('[:-]').tolist(), columns=['Chromosome', 'Start', 'End'])
    peak_names['Start'] = peak_names['Start'].astype(int)
    peak_names['End'] = peak_names['End'].astype(int)
    return pr(peak_names).sort().df

peaks = get_peak_from_snapatac(atac)
# %%
def get_peak_acpm_for_cell_type(atac: snap.AnnData, cell_type: str):
    """
    Get the peak acpm for a given cell type.
    """
    peaks = get_peak_from_snapatac(atac)
    counts = np.array(atac.X[atac.obs.cell_type == cell_type].sum(0)).flatten()
    acpm = np.log10(counts / counts.sum() * 1e5 + 1)
    peaks['aCPM'] = acpm/acpm.max()
    return peaks.query('Chromosome.str.startswith("chr") & ~Chromosome.str.endswith("M")')
# %%
"""
## 4. Generate Training Data Files
For each abundant cell type, we'll generate:
1. Peak accessibility file (.atac.bed) containing:
   - Peak coordinates (chr, start, end)
   - Normalized accessibility scores (aCPM)
2. Gene expression file (.rna.csv) containing:
   - Gene names
   - Normalized expression values (TPM)
"""
# %%
for cell_type in celltype_for_modeling:
    peaks = get_peak_acpm_for_cell_type(atac, cell_type)
    peaks.to_csv(f'{cell_type.replace(" ", "_").lower()}.atac.bed', sep='\t', index=False, header=False)
# %%
def get_rna_for_cell_type(rna: snap.AnnData, cell_type: str):
    """
    Get the rna for a given cell type.
    """
    counts = rna.X[rna.obs.cell_type == cell_type].sum(0)
    counts = np.log10(counts / counts.sum() * 1e6 + 1)
    counts = np.array(counts).flatten()
    rna_tpm = pd.DataFrame(counts, columns=['TPM'])
    rna_tpm['gene_name'] = rna.var.index
    return rna_tpm[['gene_name', 'TPM']].sort_values(by='gene_name', ascending=True)
# %% [markdown]
# The demo data contains only 3,000 variable genes. when you use your own data, make sure to use raw counts of all genes.
#%%
for cell_type in celltype_for_modeling:
    rna_tpm = get_rna_for_cell_type(rna_original, cell_type)
    rna_tpm.to_csv(f'{cell_type.replace(" ", "_").lower()}.rna.csv', index=False)
# %%

