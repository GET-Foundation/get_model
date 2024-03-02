
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
    peak_signal_track, peak_sequence, sample_metadata, celltype_peaks, motif_mean_std, peak_labels, hic_matrix  = zip(*batch)
    celltype_peaks = list(celltype_peaks)
    peak_signal_track = list(peak_signal_track)
    peak_sequence = list(peak_sequence)
    sample_metadata = list(sample_metadata)
    motif_mean_std = list(motif_mean_std)
    peak_labels = list(peak_labels)
    hic_matrix = list(hic_matrix)
    
    batch_size = len(celltype_peaks)
    mask_ratio = sample_metadata[0]['mask_ratio']

    n_peak_max = max([len(x) for x in celltype_peaks])
    # calculate max length of the sample sequence using peak coordinates, padding is 100 per peak
    sample_len_max = max([(x[:,1]-x[:,0]).sum()+100*x.shape[0] for x in celltype_peaks])
    peak_signal_track_boundary = []
    peak_sequence_boundary = []
    # pad each peaks in the end with 0
    for i in range(len(celltype_peaks)):
        celltype_peaks[i] = np.pad(celltype_peaks[i], ((0, n_peak_max - len(celltype_peaks[i])), (0,0)))
        # pad each track in the end with 0 which is csr_matrix, use sparse operation
        peak_signal_track[i].resize((sample_len_max, peak_signal_track[i].shape[1]))
        peak_sequence[i].resize((sample_len_max, peak_sequence[i].shape[1]))
        peak_signal_track[i] = peak_signal_track[i].todense()
        peak_sequence[i] = peak_sequence[i].todense()
        cov = (celltype_peaks[i][:,1]-celltype_peaks[i][:,0]).sum()
        real_cov = peak_signal_track[i].sum()
        conv = 50#int(min(500, max(100, int(cov/(real_cov+20)))))
        peak_signal_track[i] = np.convolve(np.array(peak_signal_track[i]).reshape(-1), np.ones(50)/50, mode='same')
        # if sample_track[i].max()>0:
        #     sample_track[i] = sample_track[i]/sample_track[i].max()

    celltype_peaks = np.stack(celltype_peaks, axis=0)
    celltype_peaks = torch.from_numpy(celltype_peaks)
    peak_signal_track = np.stack(peak_signal_track, axis=0)
    peak_signal_track = torch.from_numpy(peak_signal_track)
    peak_sequence = np.hstack(peak_sequence)
    peak_sequence = torch.from_numpy(peak_sequence).view(-1, batch_size, 4)
    peak_sequence = peak_sequence.transpose(0,1)
    motif_mean_std = np.stack(motif_mean_std, axis=0)
    motif_mean_std = torch.FloatTensor(motif_mean_std)
    hic_matrix = np.stack(hic_matrix, axis=0)
    hic_matrix = torch.FloatTensor(hic_matrix)
    peak_len = celltype_peaks[:,:,1]-celltype_peaks[:,:,0]
    padded_peak_len = peak_len + 100
    total_peak_len = peak_len.sum(1)
    n_peaks = (peak_len>0).sum(1)
    # max_n_peaks = n_peaks.max()
    max_n_peaks = n_peak_max
    peak_peadding_len = n_peaks*100
    tail_len = peak_sequence.shape[1] - peak_peadding_len - peak_len.sum(1)
    # flatten the list
    chunk_size = torch.cat([torch.cat([padded_peak_len[i][0:n],tail_len[i].unsqueeze(0)]) for i, n in enumerate(n_peaks)]).tolist()

    mask = torch.stack([torch.cat([torch.zeros(i), torch.zeros(max_n_peaks-i)-10000]) for i in n_peaks.tolist()])
    maskable_pos = (mask+10000).nonzero()

    for i in range(batch_size):
        maskable_pos_i = maskable_pos[maskable_pos[:,0]==i,1]
        idx = np.random.choice(maskable_pos_i, size=np.ceil(mask_ratio*len(maskable_pos_i)).astype(int), replace=False)
        mask[i,idx] = 1
    
    if peak_labels[0] is not None:
        # pad each element to max_n_peaks using zeros
        for i in range(len(peak_labels)):
            peak_labels[i] = np.pad(peak_labels[i], ((0, max_n_peaks - len(peak_labels[i])), (0,0)))
        peak_labels = np.stack(peak_labels, axis=0)
        # if aTPM < 0.1, set the expression to 0
        n_peak_labels = peak_labels.shape[-1]
        if n_peak_labels >= 3:
            # assuming the third column is aTPM, use aTPM to thresholding the expression
            peak_labels = peak_labels.reshape(-1, n_peak_labels)
            peak_labels[peak_labels[:,2]<0.1, 0] = 0
            peak_labels[peak_labels[:,2]<0.1, 1] = 0
            peak_labels = peak_labels.reshape(batch_size, -1, n_peak_labels)
            other_peak_labels = peak_labels[:,:,2:]
            exp_label = peak_labels[:,:,0:2]
        exp_label = torch.from_numpy(exp_label) # B, R, C=2 RNA+,RNA-,ATAC
        other_peak_labels = torch.from_numpy(other_peak_labels)
    else:
        exp_label = 0
        peak_labels = 0
        other_peak_labels = 0

    return peak_signal_track, peak_sequence, sample_metadata, celltype_peaks, peak_signal_track_boundary, peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, exp_label, other_peak_labels, hic_matrix
