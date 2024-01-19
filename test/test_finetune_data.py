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

# %%
cdz = CelltypeDenseZarrIO('/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr')
# %%
cdz = cdz.subset_celltypes_with_data_name()
#%%
cdz = cdz.leave_out_celltypes_with_pattern('Astrocyte')



#%%
pretrain = PretrainDataset(['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr',
                            ],
                           '/pmglocal/xf2217/get_data/hg38.zarr', 
                           '/pmglocal/xf2217/get_data/hg38_motif_result.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue_open_exp', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=1000, n_peaks_lower_bound=50, n_peaks_upper_bound=100, leave_out_celltypes='Enterocyte', leave_out_chromosomes='chr11', is_train=False, additional_peak_columns=['Expression_positive', 'Expression_negative'], non_redundant='max_depth', use_insulation=False)
pretrain.__len__()
# %%
pretrain.datapool.insulation
# %%
list(pretrain.datapool.peaks_dict.keys())
# %%
pretrain.preload_data_packs = ['']
pretrain.reload_data(0)

# %%
pretrain.__getitem__(10)
# %%
from get_model.dataset.zarr_dataset import worker_init_fn_get
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=8,
    num_workers=64,
    pin_memory=True,
    drop_last=True,
    collate_fn=get_rev_collate_fn,
    worker_init_fn=worker_init_fn_get,
)

# %%
from get_model.model.model import GETFinetune  
from get_model.utils import load_state_dict
import torch.nn as nn
loss_masked = nn.PoissonNLLLoss(log_input=False, reduce='mean')
model = GETFinetune(
        num_regions=100,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        flash_attn=True,
        nhead=12,
        dropout=0.1,
        output_dim=2,
        pos_emb_components=[],
    )
#%%
checkpoint = torch.load('/pmglocal/xf2217/output_rev_from_scratch_ATACSplitPool_unnorm_finetune_fetal_Erythroblast_leaveout_chr_bidirectional/checkpoint-135.pth')
#%%
model.load_state_dict(checkpoint["model"], strict=False)

#%%
model.eval()
model.cuda()
#%%
for i, j in model.atac_attention.joint_conv.named_parameters():
    print(i, j.requires_grad)
    weight = j.detach().cpu().numpy()
#%%
import matplotlib.pyplot as plt
plt.imshow(weight[:,0,:])
#%%
# plot as six line plot
for i in range(6):
    plt.imshow(weight[i,0:100,:])
#%%
# %%
losses = []
xs = []
ys = []
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i, batch in tqdm(enumerate(data_loader_train)):
        if i > 100:
            break
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels = batch
        if min(chunk_size)<0:
            continue

        bool_mask_pos = mask.clone()
        bool_mask_pos[bool_mask_pos != -10000] = 1
        bool_mask_pos[bool_mask_pos != 1] = 0
        bool_mask_pos = bool_mask_pos.bool().to('cuda', non_blocking=True).unsqueeze(-1)
        peak_seq = peak_seq.bfloat16().cuda()
        sample_track = sample_track.bfloat16().cuda()
        labels = labels.bfloat16().cuda()
        atac, exp = model.forward(peak_seq, sample_track, bool_mask_pos, chunk_size, n_peaks.cuda(), max_n_peaks, motif_mean_std.cuda())
        # mask_for_loss = mask.clone()

        # mask_for_loss[mask_for_loss!=1]=0
        exp = exp * bool_mask_pos
        indices = torch.where(bool_mask_pos==1)
        exp = exp[indices[0], indices[1], :].flatten()
        exp_target = labels * bool_mask_pos
        exp_target = exp_target[indices[0], indices[1], :].flatten()
        loss_masked_value = loss_masked(exp , exp_target)
        #loss_atac_value = loss_atac(atac, labels_atac)
        # print(loss_masked_value, loss_atac_value) # masked loss is around 5 times larger than atac loss
        loss = loss_masked_value.item() #+ loss_atac_value * 5
        losses.append(loss)
        x = exp.float().detach().cpu().numpy().flatten()
        y = exp_target.float().detach().cpu().numpy().flatten()
        x = x[y>0]
        y = y[y>0]
        xs.append(x.flatten())
        ys.append(y.flatten())
xs = np.concatenate(xs).flatten()
ys = np.concatenate(ys).flatten()
# %%

sns.scatterplot(x=ys, y=xs, s=2, alpha=1)
# add correlation as text
corr = np.corrcoef(xs, ys)[0,1]
corr = round(corr, 2)
plt.title(f'Correlation: {corr}')
# %%
np.mean(losses)
# %%
# r^2
from sklearn.metrics import r2_score
r2_score(ys, xs)
# %%
