
import numpy as np
import torch

def sparse_coo_to_tensor(coo):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)
 
    return torch.sparse_coo_tensor(i, v, s)


def sparse_batch_collate(batch: list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, targets_batch, cells_batch, ctcf_pos = zip(*batch)
    data_batch = torch.stack(
        [sparse_coo_to_tensor(data) for data in data_batch]
    ).to_dense()
    targets_batch = torch.stack(
        [sparse_coo_to_tensor(data) for data in targets_batch]
    ).to_dense()
    ctcf_pos_batch = torch.FloatTensor(np.stack(ctcf_pos))
    return data_batch, targets_batch, cells_batch, ctcf_pos_batch


def get_rev_collate_fn(batch):
    # zip and convert to list
    sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, motif_mean_std = zip(*batch)
    celltype_peaks = list(celltype_peaks)
    sample_track = list(sample_track)
    sample_peak_sequence = list(sample_peak_sequence)
    sample_metadata = list(sample_metadata)
    motif_mean_std = list(motif_mean_std)
    batch_size = len(celltype_peaks)
    mask_ratio = sample_metadata[0]['mask_ratio']

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
        cov = (celltype_peaks[i][:,1]-celltype_peaks[i][:,0]).sum()
        real_cov = sample_track[i].sum()
        conv = int(min(500, max(100, int(cov/(real_cov+20)))))
        sample_track[i] = np.convolve(np.array(sample_track[i]).reshape(-1), np.ones(conv), mode='same')
        if sample_track[i].max()>0:
            sample_track[i] = sample_track[i]/sample_track[i].max()

    celltype_peaks = np.stack(celltype_peaks, axis=0)
    celltype_peaks = torch.from_numpy(celltype_peaks)
    sample_track = np.stack(sample_track, axis=0)
    sample_track = torch.from_numpy(sample_track)
    sample_peak_sequence = np.hstack(sample_peak_sequence)
    sample_peak_sequence = torch.from_numpy(sample_peak_sequence).view(-1, batch_size, 4)
    sample_peak_sequence = sample_peak_sequence.transpose(0,1)
    motif_mean_std = np.stack(motif_mean_std, axis=0)
    motif_mean_std = torch.FloatTensor(motif_mean_std)
    peak_len = celltype_peaks[:,:,1]-celltype_peaks[:,:,0]
    padded_peak_len = peak_len + 100
    total_peak_len = peak_len.sum(1)
    n_peaks = (peak_len>0).sum(1)
    # max_n_peaks = n_peaks.max()
    max_n_peaks = 200
    peak_peadding_len = n_peaks*100
    tail_len = sample_peak_sequence.shape[1] - peak_peadding_len - peak_len.sum(1)
    # flatten the list
    chunk_size = torch.cat([torch.cat([padded_peak_len[i][0:n],tail_len[i].unsqueeze(0)]) for i, n in enumerate(n_peaks)]).tolist()

    mask = torch.stack([torch.cat([torch.zeros(i), torch.zeros(max_n_peaks-i)-10000]) for i in n_peaks.tolist()])
    maskable_pos = (mask+10000).nonzero()

    for i in range(batch_size):
        maskable_pos_i = maskable_pos[maskable_pos[:,0]==i,1]
        idx = np.random.choice(maskable_pos_i, size=np.ceil(mask_ratio*len(maskable_pos_i)).astype(int), replace=False)
        mask[i,idx] = 1

    

    return sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std