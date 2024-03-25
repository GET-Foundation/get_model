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
from get_model.engine import train_chrombpnet, train_class_batch
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
import h5py
# load bias model
h5 = h5py.File('/pmglocal/xf2217/THP_data/chrombpnet_output/models/chrombpnet.h5', 'a')
#%%
# h5['model_weights']['bpnet_1conv']['bpnet_1conv']['kernel:0'][:]
# for every weights, overwrite the weights in the model to zero
for key in h5['model_weights']:
    for subkey in h5['model_weights'][key]:
        print(subkey,h5['model_weights'][key][subkey]['kernel:0'][:].shape)
        # h5['model_weights'][key][subkey]['kernel:0'][:] = 0
        # h5['model_weights'][key][subkey]['bias:0'][:] = 0
#%%
# wandb.login()
# run = wandb.init(
#     project="get",
#     name="finetune-gbm",
# )

pretrain = PretrainDataset(zarr_dirs=['/pmglocal/xf2217/get_data/encode_hg38atac_dense.zarr',
                            ],
                           genome_seq_zarr='/pmglocal/xf2217/get_data/hg38.zarr', 
                           genome_motif_zarr='/pmglocal/xf2217/get_data/hg38_motif_result.zarr', insulation_paths=[
                           '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_p0.01_tissue_open_exp', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=1000, n_peaks_lower_bound=10, insulation_subsample_ratio=0.8, n_peaks_upper_bound=100, leave_out_celltypes=None, leave_out_chromosomes=None, is_train=False, additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], non_redundant=None, use_insulation=False, dataset_size=400)
pretrain.__len__()
#%%
pretrain.datapool._load_peaks()
#%%
data = pretrain.debug_getitem(0)
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
model.eval()
model.cuda()
#%%
criterion = nn.PoissonNLLLoss(log_input=False, reduce='mean')
#%%
losses = []
preds = []
obs = []
preds_atac = []
obs_atac = []
tss_atac = []
confidence_scores = []
confidence_target_scores = []

with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i, batch in tqdm(enumerate(data_loader_train)):
        if i > 200:
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
        results = train_chrombpnet(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std
        , atac_targets, labels_data,  other_labels, criterion)

# %%
result = train_chrombpnet(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_targets, labels_data,  other_labels, criterion)
# %%
results
# %%
