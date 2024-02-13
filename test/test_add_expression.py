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
from caesar.io.gencode import Gencode
gencode = Gencode()

#%%
for id in tqdm(np.array(cdz.ids)[np.where(pd.Series(cdz.ids).str.contains('Astrocyte'))[0]]):
    if id.split('.')[0] not in cell_annot_dict:
        continue
    exp_id = cell_annot_dict[id.split('.')[0]]
    exp = pd.read_feather(
        f'/manitou/pmg/users/xf2217/pretrain_human_bingren_shendure_apr2023/fetal_adult/{exp_id}.promoter_exp.feather'
    )
    exp = pd.merge(gencode.gtf, exp[['gene_id', 'TPM']].drop_duplicates(), left_on='gene_id', right_on='gene_id')
    exp['Strand'] = exp.Strand.map({'+':0, '-':1})

    peaks = cdz.get_peaks(id, name='peaks_q0.01_tissue_open')
    overlap = pr(peaks.reset_index(drop=True).reset_index()).join(pr(exp).extend(300), suffix="_exp", how='left').df[['index', 'Chromosome', 'Start', 'End', 'gene_name', 'Strand', 'TPM']].drop_duplicates()
    print(overlap)
    row_col_data = overlap.query('gene_name!="-1" & TPM>=0').groupby(['index', 'Strand']).TPM.max().fillna(0).reset_index().values
    row, col, data = row_col_data[:,0], row_col_data[:,1], row_col_data[:,2]
    row = row.astype(np.int64)
    col = col.astype(np.int64)
    data = data.astype(np.float32)
    exp_array = csr_matrix((data, (row, col)), shape=(len(peaks), 2)).todense()
    peaks = peaks.reset_index(drop=True)
    peaks['TSS'] = 0
    peaks['Expression_positive'] = exp_array[:,0]
    peaks['Expression_negative'] = exp_array[:,1]
    peaks['TSS'][np.unique(row)] = 1 
    peaks['aTPM'] = np.log10(peaks.Count / peaks.Count.sum() * 1e5 + 1)
    peaks['aTPM'] = peaks['aTPM'] / peaks['aTPM'].max()
    peaks.loc[peaks.aTPM<0.1, 'Expression_negative'] = 0
    peaks.loc[peaks.aTPM<0.1, 'Expression_positive'] = 0
    cdz.write_peaks(id, 
                peaks,
                'peaks_q0.01_tissue_open_exp',
                ['Start', 'End', 'Expression_positive', 'Expression_negative', 'aTPM', 'Count', 'TSS'],
                overwrite=True)

# %%
cdz = CelltypeDenseZarrIO(
    '/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr', 'r'
)
# %%
cdz = cdz.subset_celltypes_with_data_name('peaks_q0.01_tissue_open')
# %%
cid = 'Fetal Astrocyte 1.shendure_fetal.sample_36_cerebrum.1024'
peaks = cdz.get_peaks(cid, 'peaks_q0.01_tissue_open_exp')
# peaks.loc[peaks.aTPM<0.1, 'Expression_negative'] = 0
# peaks.loc[peaks.aTPM<0.1, 'Expression_positive'] = 0
#%%
accessibility = cdz.get_peak_counts(cid, 'peaks_q0.01_tissue_open_exp')
#%%
peaks['Exp'] = peaks['Expression_positive'] + peaks['Expression_negative']
#%%
peaks['aTPM'] = np.log10(peaks['Count']/peaks['Count'].sum()*1e5+1)
peaks['aTPM'] = peaks['aTPM']/peaks['aTPM'].max()
#%%
peaks['logCount'] = np.log10(peaks['Count']+1)
#%%
peaks['Exp'] = peaks['Expression_positive'] + peaks['Expression_negative']
# peaks.loc[peaks.aTPM<0.2, 'Exp'] = 0
peaks.query('TSS==1').plot(x='Exp', y='aTPM', kind='scatter', s=0.15)
#%%
from sklearn.metrics import r2_score
r2_score(peaks.query('Expression_negative>0')['Expression_negative'], peaks.query('Expression_negative>0')['aTPM'])
#%%
peaks.query('TSS==1 & Exp>0')[['Exp','aTPM']].corr()
# %%
# pearson correlation
from scipy.stats import pearsonr
pearsonr(peaks.query('Exp>0')['Exp'], peaks.query('Exp>0')['aTPM'])

# %%
track = cdz.get_track('Fetal Erythroblast 1.shendure_fetal.sample_35_liver.4096', 'chr19', 54189157, 54191526) 
import seaborn as sns
# conv with 100 bp 
track = np.convolve(track, np.ones(50)/50, mode='same')
sns.lineplot(x=np.arange(track.shape[0]), y=track)
# %%
peaks.query('Chromosome=="chr1" & End<634339')
# %%
liver_1_peaks_hg38 = pd.read_csv('/burg/home/xf2217/liver_1/liver_1.hg38.bed', sep='\t', header=None, names=['Chromosome', 'Start', 'End', 'Exp_0', 'Exp_1', 'aTPM'])
# %%
peaks
# %%
liver_1_with_zarr = pr(liver_1_peaks_hg38).join(pr(peaks), suffix='_zarr').df
# %%
liver_1_with_zarr.query('aTPM_zarr>0.05').plot(x='Exp_1', y='Expression_negative', kind='scatter', s=0.15)
# %%
liver_1_with_zarr.query('(Exp_0>0 & Expression_positive==0) & aTPM_zarr>0.1')
# %%
