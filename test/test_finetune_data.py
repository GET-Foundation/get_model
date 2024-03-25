#%%
import numpy as np
# import seaborn as sns
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm
import wandb
import torch.nn as nn
import matplotlib.pyplot as plt
from get_model.dataset.zarr_dataset import get_padding_pos
from get_model.engine import train_class_batch
from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack, CelltypeDenseZarrIO, worker_init_fn_get
from get_model.model.model import GETFinetune, GETFinetuneExpATAC, GETFinetuneExpATACFromSequence, GETFinetuneChrombpNet
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

pretrain = PretrainDataset(zarr_dirs=['/pmglocal/xf2217/get_data/encode_hg38atac_dense.zarr',
                            ],
                           genome_seq_zarr='/pmglocal/xf2217/get_data/hg38.zarr', 
                           genome_motif_zarr='/pmglocal/xf2217/get_data/hg38_motif_result.zarr', insulation_paths=[
                           '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue_open_exp', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=2114, n_peaks_lower_bound=2, insulation_subsample_ratio=0.8, n_peaks_upper_bound=10, leave_out_celltypes='k562', leave_out_chromosomes=['chr4','chr14'], is_train=False, additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], non_redundant=None, use_insulation=False, dataset_size=10000, random_shift_peak=False)
pretrain.__len__()
#%%
pretrain.debug_getitem(0)

#%%
# import pandas as pd
# pd.concat(dfs).sort_values('aTPM').query('aTPM>0.1')['sample'].str.split('.').str[2].str.split('_').str[0].unique()
#%%
import pandas as pd
df  = pretrain.datapool.peaks_dict['k562.encode_hg38atac.ENCFF128WZG.max']
df_ = pd.concat([df[['Expression_positive', 'aTPM', 'TSS']].rename({'Expression_positive':'Exp'}, axis=1),df[['Expression_negative', 'aTPM', 'TSS']].rename({'Expression_negative':'Exp'}, axis=1)]).query('TSS==1')
#%%
pretrain.datapool.load_data(
    data_key='encode_hg38atac_dense.zarr', 
    celltype_id='k562.encode_hg38atac.ENCFF128WZG.max',
    chr_name='chr8',
    start=129633446-1000000,
    end=129633446+1000000)
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
np.corrcoef(output_.Exp, output_.aTPM/output_.aTPM.mean()*output_.Exp.mean())
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
model = GETFinetuneExpATACFromSequence(
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
        # atac_kernel_num = 161,
        # atac_kernel_size = 3,
        # joint_kernel_num = 161,
        # final_bn = True,
    )

model = GETFinetuneChrombpNet(
        num_regions=100,
        motif_prior=False,
        embed_dim=768,
        num_layers=7,
        d_model=768,
        nhead=12,
        dropout=0.1,
        output_dim=2,
    
)
#%%
checkpoint = torch.load('/pmglocal/xf2217/Expression_Finetune_k562.Chr4&14.conv20.chrombpnet.shift10.R100L2114.augmented.20240308/checkpoint-40.pth')
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

#%%
criterion = nn.PoissonNLLLoss(log_input=False, reduce='mean')
#%%
losses = []
preds = []
obs = []
preds_atac = []
obs_atac = []
tss_atac = []
preds_atpm = []
obs_atpm = []
preds_aprofile = []
obs_aprofile = []

confidence_scores = []
confidence_target_scores = []
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i, batch in tqdm(enumerate(data_loader_train)):
        if i > 10000:
            break
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels, hic_matrix = batch
        if min(chunk_size)<0:
            continue
        device = 'cuda'
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        # chunk_size = chunk_size.to(device, non_blocking=True)
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()
        other_labels = other_labels.to(device, non_blocking=True).bfloat16()
        # compute output
        atac_targets = other_labels[:,:,0]
        result = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std
        , atac_targets, labels_data,  other_labels, criterion)
        
        if model._get_name() =='GETFinetuneChrombpNet' or args.model=='get_finetune_motif_chrombpnet':
            loss = result['loss'].item()
            loss_atpm = result['loss_atpm'].item()
            loss_aprofile = result['loss_aprofile'].item()
            atpm = result['atpm_pred']
            atpm_target = result['atpm_target']
            aprofile = result['aprofile_pred']
            aprofile_target = result['aprofile_target']
            preds_atpm.append(atpm.float().reshape(-1).detach().cpu().numpy())
            obs_atpm.append(atpm_target.float().reshape(-1).detach().cpu().numpy())
            preds_aprofile.append(aprofile.float().reshape(-1).detach().cpu().numpy())
            obs_aprofile.append(aprofile_target.float().reshape(-1).detach().cpu().numpy())

        else:
            exp = result['exp_pred']
            exp_target = result['exp_target']
            atac = result['atac_pred']
            atac_target = result['atac_target']
            loss = result['loss'].item()
            loss_atac = result['loss_atac'].item()
            loss_exp = result['loss_exp'].item()
            loss_confidence = result['loss_confidence'].item()
            padding_mask = get_padding_pos(mask)
            mask_for_loss = 1-padding_mask
            padding_mask = padding_mask.to(device, non_blocking=True).bool()
            mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
            indices = torch.where(mask_for_loss==1)
            # other_labels is B, R, N where [:,:, 1] is TSS indicator
            other_labels_reshape = other_labels[indices[0], indices[1], 1].flatten()
            preds.append(exp.reshape(-1, 2)[other_labels_reshape==1, :].reshape(-1).detach().cpu().numpy())
            obs.append(exp_target.reshape(-1,2)[other_labels_reshape==1, :].reshape(-1).detach().cpu().numpy())
            preds_atac.append(atac.reshape(-1).detach().cpu().numpy())
            obs_atac.append(atac_target.reshape(-1).detach().cpu().numpy())


    if model._get_name()=='GETFinetuneChrombpNet':
        preds_atpm = np.concatenate(preds_atpm, axis=0).reshape(-1)
        obs_atpm = np.concatenate(obs_atpm, axis=0).reshape(-1)
        preds_aprofile = np.concatenate(preds_aprofile, axis=0).reshape(-1)
        obs_aprofile = np.concatenate(obs_aprofile, axis=0).reshape(-1)
        bin=100
        obs_aprofile = np.array([np.mean(obs_aprofile[i:i+bin]) for i in range(0, len(obs_aprofile), bin)])
        preds_aprofile = np.array([np.mean(preds_aprofile[i:i+bin]) for i in range(0, len(preds_aprofile), bin)])
        # r2score_atpm, pearsonr_score_atpm, spearmanr_score_atpm = cal_score_stats(preds_atpm, obs_atpm, data_loader, args)
        # r2score_aprofile, pearsonr_score_aprofile, spearmanr_score_aprofile = cal_score_stats(preds_aprofile, obs_aprofile, data_loader, args)

        # preds.append(exp.reshape(-1).detach().cpu().numpy())
        # obs.append(exp_target.reshape(-1).detach().cpu().numpy())

# %%
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

def plot_scatter_with_limits(preds, obs, hue, xlim=None, ylim=None, figsize=(5, 5), title=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.scatterplot(x=obs, y=preds, ax=ax, s=3, alpha=0.3, hue=hue)
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    corr = pearsonr(preds, obs)[0]
    spearman = spearmanr(preds, obs)[0]
    r2 = r2_score(obs, preds)
    ax.text(0.2, 0.8, f'Pearson r={corr:.2f}\nR2={r2:.2f}\nSpearman r={spearman:.2f}'
            , ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')

    if title is not None:
        ax.set_title(title)
    return fig, ax

#%%
plot_scatter_with_limits(preds_atpm, obs_atpm, None, xlim=(0, 1), ylim=(0, 1), title='aTPM')
#%%
plot_scatter_with_limits(preds_aprofile[0:10000], obs_aprofile[0:10000], None, xlim=(0, 12), ylim=(0, 12), title='aProfile')

# %%
