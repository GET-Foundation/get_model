# %%
from scipy.sparse import load_npz
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset
from get_model.dataset.zarr_dataset import ReferenceRegionMotif, ReferenceRegionMotifConfig, _chromosome_splitter
from get_model.dataset.zarr_dataset import PretrainDataset, InferenceDataset, ReferenceRegionDataset
from scipy.sparse import coo_matrix
cfg = ReferenceRegionMotifConfig()
rrm = ReferenceRegionMotif(cfg)
# %%

pretrain = PretrainDataset(zarr_dirs=['/home/xf2217/Projects/get_data/joung_tfatlas_dense.zarr',
                                      ],
                           genome_seq_zarr='/home/xf2217/Projects/get_data/hg38.zarr',
                           genome_motif_zarr='/home/xf2217/Projects/get_data/hg38_motif_result.zarr', insulation_paths=[
                           '/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/home/xf2217/Projects/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='fetal_tfatlas_peaks_tissue_open_exp', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=0, n_peaks_lower_bound=1, insulation_subsample_ratio=0.8, n_peaks_upper_bound=900, keep_celltypes='0.joung_tfatlas.shareseq.8192', leave_out_chromosomes=['chr4', 'chr14'], is_train=True, additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], non_redundant=None, use_insulation=False, dataset_size=10000, random_shift_peak=False, hic_path='/home/xf2217/Projects/geneformer_nat/data/H1_ESC.hic')
pretrain.__len__()


# %%
rrd = ReferenceRegionDataset(rrm, pretrain, quantitative_atac=True)
# %%
d = rrd.__getitem__(0)
# %%
sns.scatterplot(x=d['region_motif'][:, -1], y=d['exp_label'].sum(1))
# %%
peaks = pretrain.datapool.peaks_dict['0.joung_tfatlas.shareseq.8192']
# %%
peaks['exp'] = peaks['Expression_positive'] + peaks['Expression_negative']
peaks['aTPM'] = np.log2(peaks['Count']/peaks['Count'].sum()*1e5+1)
peaks.query('TSS==1').plot(y='exp', x='aTPM', s=1, kind='scatter')
# %%

# %%
rrm.peaks
# %%
peaks['Name'] = peaks['Chromosome'] + ':' + \
    peaks['Start'].astype(str) + '-' + peaks['End'].astype(str)
# %%
peaks['Name'].isin(rrm.peaks['peak_names']).shape
# %%
df = peaks.query('Chromosome=="chr10" & TSS==1')[
    ['aTPM', 'Expression_positive', 'Expression_negative']]
# %%
d = np.concatenate([df[['aTPM', 'Expression_positive']].values,
                   df[['aTPM', 'Expression_negative']].values], 0).T
np.corrcoef(d[0], d[1])
# %%
cerebrum_4 = load_npz(
    '/home/xf2217/Projects/new_finetune_data_all/fetal/cerebrum_4.watac.npz')
# %%
cerebrum_4[0].toarray()
# %%
d = rrm.data
# %%


def get_cutoff(d):
    # for each column, find 90% quantile value and return a vector of cutoffs
    cutoffs = []
    for i in range(d.shape[1]):
        cutoffs.append(np.quantile(d[:, i], 0.9))
    return cutoffs


d = d * (d > get_cutoff(d))
# %%
d = d/d.max(0)
# %%
sns.scatterplot(x=d[0][0:282],
                y=cerebrum_4[0].toarray().flatten()[0:282])
# %%
cfg1 = ReferenceRegionMotifConfig(
    data='/home/xf2217/Projects/get_data/fetal_union_peak_motif_v1.hg38.zarr')
rrm1 = ReferenceRegionMotif(cfg1)
# %%
rrm1.peaks
# %%
rrm.peaks
# %%
rrm1.data[0:2]
# %%
rrm
# %%
rrd = ReferenceRegionDataset(rrm, pretrain, quantitative_atac=True)
rrd1 = ReferenceRegionDataset(rrm1, pretrain, quantitative_atac=True)
# %%
d = rrd.__getitem__(0)
d1 = rrd1.__getitem__(0)
# %%
d
# %%
sns.scatterplot(x=rrm.data[347308], y=rrm1.data[0])
# %%
sns.scatterplot(x=d['region_motif'][2], y=d1['region_motif'][2])
# %%
