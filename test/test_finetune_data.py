#%%
import numpy as np
# import seaborn as sns
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm
import wandb
import torch.nn as nn
import matplotlib.pyplot as plt

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack, CelltypeDenseZarrIO, worker_init_fn_get
from get_model.model.model import GETFinetune, GETFinetuneExpATAC
#%%
# # %%
# cdz = CelltypeDenseZarrIO('/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr')
# # %%
# cdz = cdz.subset_celltypes_with_data_name()
# #%%
# cdz = cdz.leave_out_celltypes_with_pattern('Astrocyte')


# wandb.login()
# run = wandb.init(
#     project="get",
#     name="finetune-gbm",
# )

pretrain = PretrainDataset(['/pmglocal/xf2217/get_data/htan_gbm_dense.zarr',
                            ],
                           '/pmglocal/xf2217/get_data/hg38.zarr', 
                           '/pmglocal/xf2217/get_data/hg38_motif_result.zarr', [
                           '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue_open_exp', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=1000, n_peaks_lower_bound=50, n_peaks_upper_bound=100, leave_out_celltypes=None, leave_out_chromosomes=['chr4','chr14'], is_train=False, additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], non_redundant=None, use_insulation=False, dataset_size=4096)
pretrain.__len__()
#  #%%
# dfs = []
# for key in pretrain.datapool.peaks_dict:
    # df = pretrain.datapool.peaks_dict[key]
    # break
    # if df.query('Chromosome=="chr8" & Start<129633446 & End>129633446').shape[0]>0:
#         df = df.query('Chromosome=="chr8" & Start<129633446 & End>129633446')
#         df['sample'] = key
#         dfs.append(df)
#%%
# import pandas as pd
# pd.concat(dfs).sort_values('aTPM').query('aTPM>0.1')['sample'].str.split('.').str[2].str.split('_').str[0].unique()
#%%
import pandas as pd
df  = pretrain.datapool.peaks_dict['Tumor.htan_gbm.C3N-01814_CPT0167860015_snATAC_GBM_Tumor.1024']
df_ = pd.concat([df[['Expression_positive', 'aTPM', 'TSS']].rename({'Expression_positive':'Exp'}, axis=1),df[['Expression_negative', 'aTPM', 'TSS']].rename({'Expression_negative':'Exp'}, axis=1)]).query('TSS==1')
#%%
# df['Exp'] = df.Expression_positive + df.Expression_negative
# df_ = df.query('TSS==1')
#%%
df_.plot(x='aTPM', y='Exp', kind='scatter', s=1)
#%%
from sklearn.metrics import r2_score
r2_score(df_.Exp, df_.aTPM)

#%%
df_[['Exp', 'aTPM']].corr()
#%%
from caesar.io.gencode import Gencode
gencode = Gencode('hg38', gtf_dir='/manitou/pmg/users/xf2217/bmms/caesar/data')
#%%
from pyranges import PyRanges as pr 
output = pr(gencode.gtf).join(pr(df))
#%%
output = output.df
output_pos  = output.query('Strand=="+" & TSS==1')[['gene_name', 'Expression_positive', 'aTPM']].rename({'Expression_positive':'Exp'}, axis=1)
output_neg  = output.query('Strand=="-" & TSS==1')[['gene_name', 'Expression_negative', 'aTPM']].rename({'Expression_negative':'Exp'}, axis=1)
output_ = pd.concat([output_pos, output_neg])
#%%
output_ = output_.groupby('gene_name').mean()
#%%
output_.plot(x='aTPM', y='Exp', kind='scatter', s=1)
#%%
r2_score(output_.Exp, output_.aTPM/output_.aTPM.mean()*output_.Exp.mean())
#%%
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=8,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
    collate_fn=get_rev_collate_fn,
    worker_init_fn=worker_init_fn_get,
)

loss_masked = nn.PoissonNLLLoss(log_input=False, reduce='mean')
#%%
model = GETFinetuneExpATAC(
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
        final_bn = False,
    )
#%%
checkpoint = torch.load('/pmglocal/xf2217/20240221.conv50.no_atac_loss.nofreeze.nodepth.R100L1000/checkpoint-99.pth')
#%%
model.load_state_dict(checkpoint["model"], strict=True)

# checkpoint = torch.load('/pmglocal/alb2281/get_ckpts/checkpoint-135.pth')
# model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
model.cuda()
#%%
# for i, j in model.atac_attention.joint_conv.named_parameters():
#     print(i, j.requires_grad)
#     weight = j.detach().cpu().numpy()

# figsize = (10, 10)
# plt.imshow(weight[160,:,:], aspect=0.01)

#%%
# plot as six line plot
# plot as a panel horizontally
# fig, ax = plt.subplots(1, 10, figsize=(15, 2))
# for i in range(10):
#     # calculate reorder index
#     from scipy.cluster.hierarchy import linkage, dendrogram
#     Z = linkage((weight[i+10,:,:]), 'ward')
#     g = dendrogram(Z, no_plot=True)
#     ax[i].imshow((weight[i+10,:,:])[np.array(g['ivl']).astype('int')], aspect=0.01)
    
#%%
from get_model.dataset.zarr_dataset import get_mask_pos, get_padding_pos

def train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, criterion,other_labels_data):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atac, exp = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels_data)
        

    exp = exp * mask_for_loss
    indices = torch.where(mask_for_loss==1)
    exp = exp[indices[0], indices[1], :].flatten()
    exp_target = exp_target * mask_for_loss
    exp_target = exp_target[indices[0], indices[1], :].flatten()
    loss_exp = criterion(exp, exp_target)
    
    if atac is not None:
        atac = atac * mask_for_loss
        indices = torch.where(mask_for_loss==1)
        atac = atac[indices[0], indices[1], :].flatten()
        atac_target = atac_target.unsqueeze(-1) * mask_for_loss
        atac_target = atac_target[indices[0], indices[1], :].flatten()
        loss_atac = criterion(atac, atac_target)
        loss = loss_exp + loss_atac #+ loss_exp
    else:
        loss = loss_exp
    return loss, atac, exp, exp_target 

losses = []
preds = []
obs = []
preds_atac = []
obs_atac = []
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i, batch in tqdm(enumerate(data_loader_train)):
        if i > 100:
            break
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels_data  = batch
        if min(chunk_size)<0:
            continue
        device  = 'cuda'
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()
        other_labels_data = other_labels_data.to(device, non_blocking=True).bfloat16()

        # compute output
        loss, atac, exp, exp_target = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, other_labels_data[:,:,0], labels_data, loss_masked,other_labels_data)

        # other_labels_data is B, R, N where [:,:, 1] is TSS indicator
        # only append tss preds and obs
        print(other_labels_data.shape)
        print(exp.shape)
        preds.append(exp.reshape(8,100,2)[other_labels_data[:,:,1]==1, :].reshape(-1).detach().cpu().numpy())
        obs.append(exp_target.reshape(8,100,2)[other_labels_data[:,:,1]==1, :].reshape(-1).detach().cpu().numpy())
        pred_atac = atac.reshape(8,100,1).reshape(-1).detach().cpu().numpy()
        ob_atac = other_labels_data[:,:,0].float().reshape(8,100,1).reshape(-1).detach().cpu().numpy()
        preds_atac.append(pred_atac)
        obs_atac.append(ob_atac)
        # preds.append(exp.reshape(-1).detach().cpu().numpy())
        # obs.append(exp_target.reshape(-1).detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    obs = np.concatenate(obs, axis=0).reshape(-1)
    preds_atac = np.concatenate(preds_atac, axis=0).reshape(-1)
    obs_atac = np.concatenate(obs_atac, axis=0).reshape(-1)
    # preds_atac = np.concatenate(preds_atac, axis=0).reshape(-1)
    # obs_atac = np.concatenate(obs_atac, axis=0).reshape(-1)
# %%
import seaborn as sns
# preds_ = preds[obs>0]
# obs_ = obs[obs>0]
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x=preds, y=obs, s=2, alpha=1, ax=ax)
# add correlation as text
# set x lim
plt.xlim([0, 4])
# set y lim
plt.ylim([0, 4])

from scipy.stats import spearmanr, pearsonr
# r2_score(preds, obs)
from sklearn.metrics import r2_score

corr = pearsonr(preds, obs)[0]
r2 = r2_score(preds, obs)
ax.text(0.5, 0.5, f'Pearson r={corr:.2f}\nR2={r2:.2f}', ha='center', va='center', transform=ax.transAxes)
# %%
# save the plot
import seaborn as sns
# preds_ = preds[obs>0]
# obs_ = obs[obs>0]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x=preds_atac, y=obs_atac, s=5, alpha=1, ax=ax)
# add correlation as text
# set x lim
# plt.xlim([0, 1])
# set y lim
# plt.ylim([0, 1])

from scipy.stats import spearmanr, pearsonr
# r2_score(preds, obs)
from sklearn.metrics import r2_score

corr = pearsonr(preds_atac, obs_atac)[0]
r2 = r2_score(preds_atac, obs_atac)
ax.text(0.5, 0.5, f'Pearson r={corr:.2f}\nR2={r2:.2f}', ha='center', va='center', transform=ax.transAxes)
# %%