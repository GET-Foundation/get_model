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

# # %%
# cdz = CelltypeDenseZarrIO('/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr')
# # %%
# cdz = cdz.subset_celltypes_with_data_name()
# #%%
# cdz = cdz.leave_out_celltypes_with_pattern('Astrocyte')



#%%
pretrain = PretrainDataset(['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr',
                            ],
                           '/pmglocal/xf2217/get_data/hg38.zarr', 
                           '/pmglocal/xf2217/get_data/hg38_motif_result.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue_open_exp', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=1000, n_peaks_lower_bound=50, n_peaks_upper_bound=100, leave_out_celltypes='Astrocyte', leave_out_chromosomes='chr11', is_train=False, additional_peak_columns=['Expression_positive', 'Expression_negative'], non_redundant='depth_2048', use_insulation=False, dataset_size=4096)
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
#%%
model = GETFinetune(
        num_regions=100,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        flash_attn=False,
        nhead=12,
        dropout=0.1,
        output_dim=2,
        pos_emb_components=[],
        atac_kernel_num = 161,
        atac_kernel_size = 3,
        joint_kernel_num = 161,
        final_bn = True,
    )
#%%
checkpoint = torch.load('/burg/home/xf2217/checkpoint.pth')
#%%
model.load_state_dict(checkpoint["model"], strict=True)

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
# plot as a panel horizontally
fig, ax = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    # calculate reorder index
    from scipy.cluster.hierarchy import linkage, dendrogram
    Z = linkage((weight[i+10,:,:]), 'ward')
    g = dendrogram(Z, no_plot=True)
    ax[i].imshow((weight[i+10,:,:])[np.array(g['ivl']).astype('int')], aspect=0.01)
    
#%%
from get_model.dataset.zarr_dataset import get_mask_pos, get_padding_pos

def train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, criterion):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atac, exp = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std)
    exp = exp * mask_for_loss
    indices = torch.where(mask_for_loss==1)
    exp = exp[indices[0], indices[1], :].flatten()
    exp_target = exp_target * mask_for_loss
    exp_target = exp_target[indices[0], indices[1], :].flatten()
    loss_exp = criterion(exp, exp_target)
    # loss = loss_atac + loss_exp
    loss = loss_exp
    return loss, atac, exp, exp_target 
    # loss_atac = criterion(atac, atac_target)
criterion = nn.PoissonNLLLoss(log_input=False, reduce='mean')
#%%
losses = []
preds = []
obs = []
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i, batch in tqdm(enumerate(data_loader_train)):
        if i > 200:
            break
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data = batch
        if min(chunk_size)<0:
            continue
        device  = 'cuda'
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()

        # compute output
        loss, atac, exp, exp_target = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std
        , None, labels_data, criterion)

        preds.append(exp.reshape(-1).detach().cpu().numpy())
        obs.append(exp_target.reshape(-1).detach().cpu().numpy())
        # preds_atac.append(atac.reshape(-1).detach().cpu().numpy())
        # obs_atac.append(atac_targets.reshape(-1).detach().cpu().numpy())

        # metric_logger.update(loss=loss.item())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    obs = np.concatenate(obs, axis=0).reshape(-1)
    # preds_atac = np.concatenate(preds_atac, axis=0).reshape(-1)
    # obs_atac = np.concatenate(obs_atac, axis=0).reshape(-1)
# %%
preds_ = preds[obs>0]
obs_ = obs[obs>0]
sns.scatterplot(x=preds_, y=obs_, s=2, alpha=1)
# add correlation as text
# set x lim
# plt.xlim([0, 4])
# set y lim
# plt.ylim([0, 4])

from scipy.stats import spearmanr, pearsonr
# r2_score(preds, obs)
from sklearn.metrics import r2_score

corr = pearsonr(preds_, obs_)[0]
plt.title(f'Correlation: {corr}, R2: {r2_score(preds_, obs_)}')
# %%
np.mean(losses)
# %%