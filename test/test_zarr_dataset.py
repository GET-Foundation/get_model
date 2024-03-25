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
pretrain = PretrainDataset(zarr_dirs=['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr',
                            ],
                           genome_seq_zarr='/pmglocal/xf2217/get_data/hg38.zarr', 
                           genome_motif_zarr='/pmglocal/xf2217/get_data/hg38_motif_result.zarr', insulation_paths=[
                           '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], , peak_name='peaks_q0.01_tissue_open_exp', preload_count=100, n_packs=1,
                           max_peak_length=5000, center_expand_target=500, n_peaks_lower_bound=10, n_peaks_upper_bound=100, use_insulation=False, leave_out_celltypes='Astrocyte', leave_out_chromosomes='chr1', is_train=False, dataset_size=65536, additional_peak_columns=None, hic_path='/burg/pmg/users/xf2217/get_data/4DNFI2TK7L2F.hic')
pretrain.__len__()
#%%
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
from get_model.model.model import GETPretrain, GETPretrainMaxNorm  
from get_model.utils import load_state_dict
import torch.nn as nn
loss_masked = nn.MSELoss()
model = GETPretrainMaxNorm(
        num_regions=100,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        flash_attn=False,
        nhead=12,
        dropout=0.1,
        output_dim=800,
        atac_kernel_num = 161,
        atac_kernel_size = 3,
        joint_kernel_num = 161,
        final_bn=False,
        joint_kernel_size = 3,
        pos_emb_components=[],
    )
#%%
checkpoint = torch.load('/pmglocal/xf2217/pretrain_conv50.maxnorm.R100L500/checkpoint-21.pth')
#%%
model.load_state_dict(checkpoint["model"], strict=False)

#%%
model.eval()
model.cuda()
#%%
for i, j in model.atac_attention.joint_conv.named_parameters():
    print(i, j.dtype)
    weight = j.detach().cpu().numpy()
    
#%%
import seaborn as sns
# row index reordered
# g.dendrogram_row.reordered_ind
#%%
# plot as a panel horizontally
fig, ax = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    # calculate reorder index
    from scipy.cluster.hierarchy import linkage, dendrogram
    Z = linkage((weight[i+10,:,:]-weight[3+10,:,:]), 'ward')
    g = dendrogram(Z, no_plot=True)
    ax[i].imshow((weight[i+20,:,:])[np.array(g['ivl']).astype('int')], aspect=0.01)
    
#%%
# compare two random weight[i].flatten() using scatter plot
i = np.random.randint(0, 161)
j = np.random.randint(0, 161)
sns.scatterplot(x=weight[i][:,1].flatten(), y=weight[j][:,1].flatten())
#%%
loss_values = []
output_masked_list = []
target_list = []
from get_model.dataset.zarr_dataset import get_mask_pos, get_padding_pos
# with torch.amp.autocast('cuda', dtype=torch.bfloat16):
for i, batch in tqdm(enumerate(data_loader_train)):
    if i > 100:
        break
    if min(batch['chunk_size'])<0:
        continue

    loss_mask = batch['loss_mask'].cuda()
    padding_mask = batch['padding_mask'].cuda()
    peak_seq = batch['sample_peak_sequence'].cuda()
    sample_track = batch['sample_track'].cuda()
    chunk_size = batch['chunk_size']
    n_peaks = batch['n_peaks'].cuda()
    max_n_peaks = batch['max_n_peaks']
    motif_mean_std = batch['motif_mean_std'].cuda()
    output_masked, _, target = model.forward(peak_seq, sample_track, loss_mask, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std)
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

    loss_mask = loss_mask.unsqueeze(-1)
    loss_masked_value = loss_masked(input=output_masked*loss_mask, target=regions_embed*loss_mask)
    #loss_atac_value = loss_atac(atac, labels_atac)
    # print(loss_masked_value, loss_atac_value) # masked loss is around 5 times larger than atac loss
    loss = loss_masked_value #+ loss_atac_value * 5
    output_masked_list.append((output_masked*loss_mask).float().detach().cpu().numpy())
    target_list.append((regions_embed*loss_mask).float().detach().cpu().numpy())
    loss_value = loss.item()
    loss_values.append(loss_value)
#%%
plt.hist(loss_values, bins=10)
#%%
output_masked_values = np.concatenate(output_masked_list, axis=0)
regions_embed_values = np.concatenate(target_list, axis=0)

#%%
# save concatenated output and target as npy
np.save('output_masked_values_rcc.npy', output_masked_values)
np.save('regions_embed_values_rcc.npy', regions_embed_values)
# %%
# load
import numpy as np
import seaborn as sns
# output_masked_values_fetal = np.load('output_masked_values_rcc.npy')#[:,:,100]#.flatten()
# regions_embed_values_fetal = np.load('regions_embed_values_rcc.npy')#[:,:,100]#.flatten()
# output_masked_values_gbm = np.load('output_masked_values_rcc.npy')[:,:,639:].flatten()
# regions_embed_values_gbm = np.load('regions_embed_values_rcc.npy')[:,:,639:].flatten()
output_masked_values = output_masked_values_gbm[regions_embed_values_gbm!=0]
regions_embed_values = regions_embed_values_gbm[regions_embed_values_gbm!=0]
#%%
# filter = (output_masked_values>0 ) & (regions_embed_values>0)
# output_masked_values = output_masked_values[filter]
# regions_embed_values = regions_embed_values[filter]
# random sample 10000 points
# idx = np.random.choice(len(output_masked_values), 100, replace=False)
# output_masked_values = output_masked_values[idx]
# regions_embed_values = regions_embed_values[idx]
from matplotlib import pyplot as plt
# kde plot with shade 
idx = 2
output_masked_values = output_masked_list[idx][:,:,0].reshape(-1)
regions_embed_values = target_list[idx][:,:,0].reshape(-1)
sns.scatterplot(x = output_masked_values, y = regions_embed_values,s=10, alpha=0.5)
# xlim and ylim 0, 0.8
# plt.xlim(0, 0.1)
# plt.ylim(0, 0.1)
# equal aspect ratio
# plt.gca().set_aspect('equal', adjustable='box')
# label R2 and pearson r
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
pearson_r, _ = pearsonr(output_masked_values, regions_embed_values)
r2 = r2_score(output_masked_values, regions_embed_values)
# plt.text(0.1, 0.1, f'Pearson r={pearson_r:.3f}; R2={r2:.3f}', fontsize=10)
# x label
plt.xlabel('Predicted', fontsize=10)
# y label
plt.ylabel('Masked Target', fontsize=10)
#%%
# idx_x = np.random.choice(len(regions_embed_values_gbm), 80, replace=False)
# idx_y = np.random.choice(len(regions_embed_values_gbm), 80, replace=False)
# g=sns.scatterplot(x=regions_embed_values_gbm[idx_x].reshape(-1, 655).std(0)/regions_embed_values_fetal[idx_y].reshape(-1, 655).std(0), y=np.arange(655))
# y range
g.set_ylim(0, 655)
g.set_xlim(0.5,1.5)
# %%
