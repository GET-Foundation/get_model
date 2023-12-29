# %%
import logging

import numpy as np
import seaborn as sns
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# %%
pretrain = PretrainDataset(['/pmglocal/xf2217/shendure_fetal/shendure_fetal_dense.zarr',
                            '/pmglocal/xf2217/bingren_adult/bingren_adult_dense.zarr'],
                           '/manitou/pmg/users/xf2217/get_model/data/hg38.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], preload_count=50, samples_per_window=150)
pretrain.__len__()
# %%
# for i in tqdm(range(100)):
pretrain.__getitem__(0)


# %%
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=64,
    num_workers=96,
    pin_memory=False,
    drop_last=True,
    collate_fn=get_rev_collate_fn
)

# %%
for batch in tqdm(data_loader_train):
    sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len = batch
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
        nhead=12,
        dropout=0.1,
        output_dim=1274,
        pos_emb_components=[],
    )
model.eval()
#%%
model.cuda()
bool_mask_pos = mask.clone()
bool_mask_pos[bool_mask_pos == -10000] = 0


a = model.forward(sample_peak_sequence.float().cuda(), sample_track.float().cuda(), bool_mask_pos.bool().cuda(), chunk_size, n_peaks.cuda(), max_n_peaks)
# %%
a[0].shape
# %%
a[2].device
# %%
