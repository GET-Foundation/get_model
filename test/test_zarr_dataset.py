#%%
import numpy as np
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset
import logging
# Setup logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
import warnings

# Suppress all deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
pretrain = PretrainDataset(['/pmglocal/xf2217/shendure_fetal/shendure_fetal_dense.zarr', 
                            '/pmglocal/xf2217/bingren_adult/bingren_adult_dense.zarr'], 
                           '/manitou/pmg/users/xf2217/get_model/data/hg38.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], preload_count=50, samples_per_window=150,
                           return_list=False)
pretrain.__len__()
# %%
# for i in tqdm(range(100)):
pretrain.__getitem__(0)

#%%



#%%
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=64,
    num_workers=96,
    pin_memory=False,
    drop_last=True,
    collate_fn = get_rev_collate_fn
)

# %%
for batch in tqdm(data_loader_train):
    sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len = batch
# %%
celltype_peaks.shape
# %%
sample_metadata
# %%
sample_peak_sequence.shape
# %%
sample_track.shape
# %%
mask






#%%
def pool(x, method='mean'):
    """
    x: (L,D)
    """
    if method == 'sum':
        return x.sum(0)
    elif method == 'max':
        return x.max(0)
    elif method == 'mean':
        return x.mean(0)


def forward(x, celltype_peaks, pool_method='sum'):
    """
    x: (batch, length, dimension)
    celltype_peaks: (batch, n_peak, 2)
    """
    # calculate the length of each peak with 100bp padding on each side

    batch, length, embed_dim = x.shape
    chunk_list = torch.split(x.reshape(-1,embed_dim), chunk_size, dim=0)
    # each element is L, D, pool the tensor
    chunk_list = torch.vstack([pool(chunk, pool_method) for chunk in chunk_list])
    # remove the padded part
    pool_idx = torch.cumsum(n_peaks+1,0)
    pool_start = torch.cat([torch.tensor(0).unsqueeze(0), pool_idx[:-1]])
    pool_end = pool_idx-1
    pool_list = [chunk_list[pool_start[i]:pool_end[i]] for i in range(len(pool_start))]
    # pad the element in pool_list if the number of peaks is not the same
    x = torch.stack([torch.cat([pool_list[i], torch.zeros(max_n_peaks-n_peaks[i], embed_dim)]) for i in range(len(pool_list))])

    return x

forward(sample_track, celltype_peaks).shape







#%%
import seaborn as sns
i=np.random.randint(0,32)
cov = (celltype_peaks[:,:,1]-celltype_peaks[:,:,0]).sum(1)
real_cov = sample_track.sum(1)
conv = (cov/real_cov).numpy().astype(np.int32)*5
# y = sample_track[i].numpy()
y = np.convolve(sample_track[i].numpy(), np.ones(conv[i]), mode='same')
sns.lineplot(y=y, x=range(len(sample_track[i].numpy())))
# %%
sample_track[i].numpy().sum()
# %%
