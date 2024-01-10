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
#%%
pretrain = PretrainDataset(['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr',
                            '/pmglocal/xf2217/get_data/bingren_adult_dense.zarr',],
                           '/pmglocal/xf2217/get_data/hg38.zarr', 
                           '/pmglocal/xf2217/get_data/hg38_motif_result.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue_open', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=1000, n_peaks_lower_bound=5, n_peaks_upper_bound=200)
pretrain.__len__()
#%%
from get_model.dataset.zarr_dataset import worker_init_fn_get
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=16,
    num_workers=64,
    pin_memory=True,
    drop_last=True,
    collate_fn=get_rev_collate_fn,
    worker_init_fn=worker_init_fn_get,
)

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
        output_dim=1280,
        pos_emb_components=[],
    )
#%%
checkpoint = torch.load('/pmglocal/xf2217/output_pretrain_rev_ATACSplitPool/checkpoint-2.pth')
#%%
model.load_state_dict(checkpoint["model"])

#%%
model.eval()
model.cuda()

#%%

loss_values = []
output_masked_list = []
target_list = []

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i, batch in tqdm(enumerate(data_loader_train)):
        if i > 10:
            break
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std = batch
        if min(chunk_size)<0:
            continue
    # for i in tqdm(range(100)):
        bool_mask_pos = mask.clone()
        bool_mask_pos[bool_mask_pos == -10000] = 0
        peak_seq = peak_seq.bfloat16().cuda()
        sample_track = sample_track.bfloat16().cuda()
        output_masked, _, target = model.forward(peak_seq, sample_track, bool_mask_pos.bool().cuda(), chunk_size, n_peaks.cuda(), max_n_peaks, motif_mean_std.cuda())
        normalize_target = False
        with torch.no_grad():
            unnorm_targets = target
            if normalize_target:
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
output_masked_values = np.concatenate(output_masked_list, axis=0)
regions_embed_values = np.concatenate(target_list, axis=0)
#%%
sns.scatterplot(x = output_masked_values.flatten(), y = np.log10(regions_embed_values.flatten()+1), s=1)
# equal aspect ratio
plt.gca().set_aspect('equal', adjustable='box')
# x lim
# x label
plt.xlabel('Output masked')
# y label
plt.ylabel('Target')
#%%