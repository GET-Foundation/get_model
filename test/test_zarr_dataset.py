#%%
import numpy as np
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from get_model.dataset.collate import csr_to_torch_sparse
from get_model.dataset.zarr_dataset import PretrainDataset
import logging
# Setup logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# %%
pretrain = PretrainDataset(['/pmglocal/xf2217/shendure_fetal/shendure_fetal_dense.zarr', 
                            '/pmglocal/xf2217/bingren_adult/bingren_adult_dense.zarr'], 
                           '/manitou/pmg/users/xf2217/get_model/data/hg38.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], preload_count=80, samples_per_window=150,
                           return_list=False)
pretrain.__len__()
# %%
# for i in tqdm(range(100)):
#     pretrain.__getitem__(0)

#%%


def collate_fn_test(batch):
    # zip and convert to list
    sample_track, sample_peak_sequence, sample_metadata, celltype_peaks = zip(*batch)
    celltype_peaks = list(celltype_peaks)
    sample_track = list(sample_track)
    sample_peak_sequence = list(sample_peak_sequence)
    sample_metadata = list(sample_metadata)
    batch_size = len(celltype_peaks)
    n_peak_max = max([len(x) for x in celltype_peaks])
    sample_len_max = max([len(x.getnnz(1)) for x in sample_peak_sequence])
    sample_track_boundary = []
    sample_peak_sequence_boundary = []
    # pad each peaks in the end with 0
    for i in range(len(celltype_peaks)):
        celltype_peaks[i] = np.pad(celltype_peaks[i], ((0, n_peak_max - len(celltype_peaks[i])), (0,0)))
        # pad each track in the end with 0 which is csr_matrix, use sparse operation
        sample_track[i].resize((sample_len_max, sample_track[i].shape[1]))
        sample_peak_sequence[i].resize((sample_len_max, sample_peak_sequence[i].shape[1]))
        sample_track[i] = sample_track[i].todense()
        sample_peak_sequence[i] = sample_peak_sequence[i].todense()

    celltype_peaks = np.stack(celltype_peaks, axis=0)
    celltype_peaks = torch.from_numpy(celltype_peaks)
    sample_track = np.hstack(sample_track).T
    sample_track = torch.from_numpy(sample_track)
    sample_peak_sequence = np.hstack(sample_peak_sequence)
    sample_peak_sequence = torch.from_numpy(sample_peak_sequence).view(-1, batch_size, 4)
    # sample_peak_sequence = sample_peak_sequence.to_sparse_csr()
    # return sample_track, sample_peak_sequence.view(-1, batch_size, 4), sample_metadata, celltype_peaks
    return sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary
#%%
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=32,
    num_workers=96,
    pin_memory=False,
    drop_last=True,
    collate_fn = collate_fn_test
)

# %%
for batch in tqdm(data_loader_train):
    sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary= batch
# %%
celltype_peaks.shape
# %%
sample_metadata
# %%
sample_peak_sequence.shape
# %%
sample_track.shape
# %%
