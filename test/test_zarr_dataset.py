# %%
import logging
from matplotlib import pyplot as plt

import numpy as np
import seaborn as sns
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack, CelltypeDenseZarrIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# %%
# zdp = ZarrDataPool(
#     ['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr'],
    
#     '/pmglocal/xf2217/get_data/hg38.zarr', 
    
#     [
#         '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather'],
#         peak_name='peaks_q0.01', max_peak_length=10000, center_expand_target=None)

# #%%
# # from get_model.dataset.zarr_dataset import PreloadDataPack
# pdp = PreloadDataPack(50, zdp)
# #%%
# while pdp.next_sample < len(pdp):
#     pdp.get_sample_with_idx(pdp.next_sample)
#     pdp.next_sample += 1
#     print(pdp.next_sample)

#%%
pretrain = PretrainDataset(['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr'],
                           '/pmglocal/xf2217/get_data/hg38.zarr', 
                           '/pmglocal/xf2217/get_data/hg38_motif_result.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue_open', preload_count=1, n_packs=1,
                           max_peak_length=5000, center_expand_target=500, n_peaks_lower_bound=5, n_peaks_upper_bound=200)
pretrain.__len__()
#%%
pretrain.__len__
#%%
for i in range(1000):
    a = pretrain.__getitem__(0)
# %%
#check maximum peak length
peak_len = []
for i, df in zdp.peaks_dict.items():
    peak_len.append(df['End']-df['Start'].values)
#%%
sns.distplot(np.concatenate(peak_len))
# %%
from get_model.dataset.zarr_dataset import worker_init_fn_get
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=8,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    collate_fn=get_rev_collate_fn,
    worker_init_fn=worker_init_fn_get,
)

# %%
for i, batch in tqdm(enumerate(data_loader_train)):
    sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std = batch
    if min(chunk_size)<0:
        continue
    # if max_n_peaks>200:
    if i > 2:
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
from get_model.utils import load_state_dict
import torch.nn as nn
loss_masked = nn.MSELoss()
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
#%%
checkpoint = torch.load('/pmglocal/xf2217/output_pretrain_rev/checkpoint-10.pth')
#%%
model.load_state_dict(checkpoint["model"])

#%%
model.eval()
model.cuda()

# #%%
# del sample_peak_sequence
# del sample_track
# del model 
# torch.cuda.empty_cache()
#%%

loss_values = []
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i, batch in tqdm(enumerate(data_loader_train)):
        if i > 500:
            break
        sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std = batch
        if min(chunk_size)<0:
            continue
    # for i in tqdm(range(100)):
        bool_mask_pos = mask.clone()
        bool_mask_pos[bool_mask_pos == -10000] = 0
        sample_peak_sequence = sample_peak_sequence.bfloat16().cuda()
        sample_track = sample_track.bfloat16().cuda()
        output_masked, _, target = model.forward(sample_peak_sequence, sample_track, bool_mask_pos.bool().cuda(), chunk_size, n_peaks.cuda(), max_n_peaks, motif_mean_std.cuda())
        normlize_target = True
        with torch.no_grad():
            unnorm_targets = target
            if normlize_target:
                regions_squeeze = unnorm_targets
                regions_norm = (
                    regions_squeeze - regions_squeeze.mean(dim=-2, keepdim=True)
                ) / (
                    regions_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt()
                    + 1e-6
                )
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                regions_embed = regions_norm
            else:
                regions_embed = unnorm_targets

            B, _, C = regions_embed.shape

        mask_for_loss = mask.clone()
        # mask_for_loss[mask_for_loss!=-10000]=1

        mask_for_loss[mask_for_loss!=1]=0
        mask_for_loss = mask_for_loss.to('cuda', non_blocking=True).unsqueeze(-1)
        loss_masked_value = loss_masked(input=output_masked*mask_for_loss, target=regions_embed*mask_for_loss)
        #loss_atac_value = loss_atac(atac, labels_atac)
        # print(loss_masked_value, loss_atac_value) # masked loss is around 5 times larger than atac loss
        loss = loss_masked_value #+ loss_atac_value * 5

        loss_value = loss.item()
        loss_values.append(loss_value)
#%%
plt.hist(loss_values, bins=20)
#%%
sns.scatterplot(x = (output_masked*mask_for_loss).float().cpu().detach().numpy().flatten(), y = (regions_embed*mask_for_loss).float().cpu().detach().numpy().flatten())
#%%
(a[2]-a[0]).mean()
#%%
sns.scatterplot(x=a[0].float().cpu().detach().numpy().flatten(), y=a[2].float().cpu().detach().numpy().flatten())
# x label
plt.xlabel('Output masked')
# y label
plt.ylabel('Target')
#%%
sns.scatterplot(x=a[0].float().cpu().detach().numpy().flatten(), y=a[2].float().cpu().detach().numpy().flatten())
# x label
plt.xlabel('Output masked')
# y label
plt.ylabel('Target')
# %%
a[0].shape
# %%
a[2].device
# %%
import pandas as pd 
import numpy as np
y = pd.read_csv('../log.txt', sep='loss', skiprows=445).iloc[:,1].str.split('(').str[0].str.split(':').str[1].astype(float)
y_conv = np.convolve(y, np.ones(20)/20, mode='valid')
import matplotlib.pyplot as plt
plt.plot(y_conv)
# %%
