#%%
import pandas as pd 
from pyranges import PyRanges as pr
import numpy as np
from scipy.sparse import csr_matrix
from caesar.io.zarr_io import CelltypeDenseZarrIO
from tqdm import tqdm
# %%
cell_annot = pd.read_csv(
    '/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/data/cell_type_pretrain_human_bingren_shendure_apr2023.txt'
).query('expression==True')
cell_annot['celltype'] = cell_annot['celltype'].str.split('+').str[0]
cell_annot_dict = cell_annot[['celltype', 'id']].set_index('celltype').to_dict()['id']
# %%
cdz = CelltypeDenseZarrIO(
    '/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr', 'a'
)
# %%
mapped_id = [cell_annot_dict[id.split('.')[0]] if id.split('.')[0] in cell_annot_dict else None for id in cdz.ids ]
mapped_id = [x for x in mapped_id if x is not None]
mapped_id = np.unique(mapped_id)

#%%
for id in tqdm(cdz.ids):
    if id.split('.')[0] not in cell_annot_dict:
        continue
    exp_id = cell_annot_dict[id.split('.')[0]]
    exp = pd.read_feather(
        f'/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/{exp_id}.promoter_exp.feather'
    )
    exp['Strand'] = exp.Strand.map({'+':0, '-':1})

    peaks = cdz.get_peaks(id, name='peaks_q0.01_tissue_open')
    overlap = pr(peaks.reset_index(drop=True).reset_index()).join(pr(exp).extend(300), suffix="_exp", how='left').df[['index', 'Chromosome', 'Start', 'End', 'gene_name', 'Strand', 'TPM']].drop_duplicates()
    row_col_data = overlap.query('gene_name!=-1 & TPM>0').groupby(['index', 'Strand']).TPM.mean().fillna(0).reset_index().values
    row, col, data = row_col_data[:,0], row_col_data[:,1], row_col_data[:,2]
    row = row.astype(np.int64)
    col = col.astype(np.int64)
    data = data.astype(np.float32)
    exp_array = csr_matrix((data, (row, col)), shape=(len(peaks), 2)).todense()
    peaks['Expression_positive'] = exp_array[:,0]
    peaks['Expression_negative'] = exp_array[:,1]
    cdz.write_peaks(id, 
                peaks,
                'peaks_q0.01_tissue_open_exp',
                ['Start', 'End', 'Expression_positive', 'Expression_negative'],
                overwrite=True)

# %%
cdz = CelltypeDenseZarrIO(
    '/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr', 'r'
)
# %%
cdz = cdz.subset_celltypes_with_data_name('peaks_q0.01_tissue_open_exp')
# %%
cid = 'Fetal Erythroblast 1.shendure_fetal.sample_40_liver.4096'
peaks = cdz.get_peaks(cid, 'peaks_q0.01_tissue_open_exp')
accessibility = cdz.get_peak_counts(cid, 'peaks_q0.01_tissue_open_exp')
peaks = pd.concat([peaks, accessibility], axis=1)
peaks['Exp'] = peaks['Expression_positive'] + peaks['Expression_negative']

#%%
peaks['aTPM'] = np.log10(peaks['Count']/peaks['Count'].sum()*1e5+1)
#%%
peaks.loc[peaks.aTPM<0.2, 'Exp'] = 0
peaks.query('Exp>0').plot(x='Exp', y='aTPM', kind='scatter', s=0.15)
# %%
# pearson correlation
from scipy.stats import pearsonr
pearsonr(peaks.query('Exp>0')['Exp'], peaks.query('Exp>0')['aTPM'])

# %%
