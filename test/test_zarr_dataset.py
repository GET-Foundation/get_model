# %%
import logging

import numpy as np
import seaborn as sns
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# %%
zdp = ZarrDataPool(
    ['/pmglocal/xf2217/shendure_fetal/shendure_fetal_dense.zarr'],
    '/manitou/pmg/users/xf2217/get_model/data/hg38.zarr', [
        '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather'],
        peak_name='peaks_p0.01', max_peak_length=10000, center_expand_target=None)

#%%
# from get_model.dataset.zarr_dataset import PreloadDataPack
pdp = PreloadDataPack(50, zdp)
#%%
while pdp.next_sample < len(pdp):
    pdp.get_sample_with_idx(pdp.next_sample)
    pdp.next_sample += 1
    print(pdp.next_sample)

#%%
pretrain = PretrainDataset(['/pmglocal/xf2217/shendure_fetal/shendure_fetal_dense.zarr',
                            '/pmglocal/xf2217/bingren_adult/bingren_adult_dense.zarr'],
                           '/manitou/pmg/users/xf2217/get_model/data/hg38.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_p0.01', preload_count=200, n_packs=2,
                           max_peak_length=5000, center_expand_target=1000)
pretrain.__len__()
# %%
#check maximum peak length
peak_len = []
for i, df in zdp.peaks_dict.items():
    peak_len.append(df['End']-df['Start'].values)
#%%
sns.distplot(np.concatenate(peak_len))
# %%
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=32,
    num_workers=16,
    pin_memory=False,
    drop_last=True,
    collate_fn=get_rev_collate_fn
)

# %%
for i, batch in tqdm(enumerate(data_loader_train)):
    if i < 100:
        continue
    sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len = batch
    if min(chunk_size)<0:
        continue
    if max_n_peaks>200:
        break
# %%
celltype_peaks.shape
# %%
sample_metadata
# %%
sample_peak_sequence.shape

# %%
sample_track.shape
# %%
from get_model.model.model import GETPretrain  
model = GETPretrain(
        num_regions=200,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        flash_attn=True,
        nhead=12,
        dropout=0.1,
        output_dim=1274,
        pos_emb_components=[],
    )
model.train()
#%%
model.cuda()
# #%%
# del sample_peak_sequence
# del sample_track
# del model 
# torch.cuda.empty_cache()
#%%
bool_mask_pos = mask.clone()
bool_mask_pos[bool_mask_pos == -10000] = 0

# to sparse
#%%
sample_peak_sequence = sample_peak_sequence.bfloat16().cuda()
sample_track = sample_track.bfloat16().cuda()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i in tqdm(range(100)):
        a = model.forward(sample_peak_sequence, sample_track, bool_mask_pos.bool().cuda(), chunk_size, n_peaks.cuda(), max_n_peaks)

#%%

#%%
(a[2]-a[0]).sum().backward()
# %%
a[0].shape
# %%
a[2].device
# %%
