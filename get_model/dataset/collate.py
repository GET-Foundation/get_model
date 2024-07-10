
from ast import List
import numpy as np
import torch

from get_model.dataset.transforms import rev_comp
from get_model.dataset.zarr_dataset import get_mask_pos, get_padding_pos
from torch.utils.data.dataloader import default_collate


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


def batch_dict_list_to_dict(batch: list):
    """
    Collate function which to transform list of dict to dict
    """
    keys = batch[0].keys()
    new_batch = {k: [d[k] for d in batch] for k in keys}
    return new_batch


def get_rev_collate_fn(batch, reverse_complement=False):
    batch = batch_dict_list_to_dict(batch)
    sample_track = list(batch['sample_track'])
    sample_peak_sequence = list(batch['sample_peak_sequence'])
    celltype_peaks = list(batch['celltype_peaks'])
    metadata = list(batch['metadata'])
    for i, meta in enumerate(metadata):
        sample_track[i] = sample_track[i]  # /meta['libsize'] * 100000000
    motif_mean_std = list(batch['motif_mean_std'])
    additional_peak_features = list(batch['additional_peak_features'])
    hic_matrix = list(batch['hic_matrix'])

    batch_size = len(celltype_peaks)
    mask_ratio = metadata[0]['mask_ratio']

    n_peak_max = max([len(x) for x in celltype_peaks])
    # calculate max length of the sample sequence using peak coordinates
    sample_len_max = max([(x[:, 1]-x[:, 0]).sum()
                         for x in celltype_peaks])
    # pad each peaks in the end with 0
    for i in range(len(celltype_peaks)):
        celltype_peaks[i] = np.pad(celltype_peaks[i], ((
            0, n_peak_max - len(celltype_peaks[i])), (0, 0)))
        # pad each track in the end with 0 which is csr_matrix, use sparse operation
        sample_track[i].resize((sample_len_max, sample_track[i].shape[1]))
        sample_peak_sequence[i].resize(
            (sample_len_max, sample_peak_sequence[i].shape[1]))
        sample_track[i] = sample_track[i].todense()
        sample_peak_sequence[i] = sample_peak_sequence[i].todense()
        # TODO Note that the rev_comp function is used here! probably not good for inference
        if reverse_complement:
            sample_peak_sequence[i], sample_track[i] = rev_comp(
                sample_peak_sequence[i], sample_track[i], prob=0.5)
        # cov = (celltype_peaks[i][:, 1]-celltype_peaks[i][:, 0]).sum()
        # real_cov = sample_track[i].sum()
        conv = 50  # int(min(500, max(100, int(cov/(real_cov+20)))))
        sample_track[i] = np.convolve(
            np.array(sample_track[i]).reshape(-1), np.ones(conv)/conv, mode='same')
        # if sample_track[i].max()>0:
        #     sample_track[i] = sample_track[i]/sample_track[i].max()

    celltype_peaks = np.stack(celltype_peaks, axis=0)
    celltype_peaks = torch.from_numpy(celltype_peaks)
    sample_track = np.stack(sample_track, axis=0)
    sample_track = torch.from_numpy(sample_track)
    sample_peak_sequence = np.hstack(sample_peak_sequence)
    sample_peak_sequence = torch.from_numpy(
        sample_peak_sequence).view(-1, batch_size, 4)
    sample_peak_sequence = sample_peak_sequence.transpose(0, 1)
    motif_mean_std = np.stack(motif_mean_std, axis=0)
    motif_mean_std = torch.FloatTensor(motif_mean_std)
    hic_matrix = np.stack(hic_matrix, axis=0)
    if hic_matrix[0] is not None:
        hic_matrix = torch.FloatTensor(hic_matrix)
    else:
        hic_matrix = torch.FloatTensor(0)
    peak_len = celltype_peaks[:, :, 1]-celltype_peaks[:, :, 0]
    total_peak_len = peak_len.sum(1)
    n_peaks = (peak_len > 0).sum(1)
    # max_n_peaks = n_peaks.max()
    max_n_peaks = n_peak_max
    tail_len = sample_peak_sequence.shape[1] - peak_len.sum(1)
    # flatten the list
    chunk_size = torch.cat([torch.cat([peak_len[i][0:n], tail_len[i].unsqueeze(
        0)]) for i, n in enumerate(n_peaks)]).tolist()

    mask = torch.stack([torch.cat([torch.zeros(i), torch.zeros(
        max_n_peaks-i)-10000]) for i in n_peaks.tolist()])
    maskable_pos = (mask+10000).nonzero()

    for i in range(batch_size):
        maskable_pos_i = maskable_pos[maskable_pos[:, 0] == i, 1]
        idx = np.random.choice(maskable_pos_i, size=np.ceil(
            mask_ratio*len(maskable_pos_i)).astype(int), replace=False)
        mask[i, idx] = 1

    if additional_peak_features[0] is not None:
        # pad each element to max_n_peaks using zeros
        for i in range(len(additional_peak_features)):
            additional_peak_features[i] = np.pad(additional_peak_features[i], ((
                0, max_n_peaks - len(additional_peak_features[i])), (0, 0)))
        additional_peak_features = np.stack(additional_peak_features, axis=0)
        # if aTPM < 0.1, set the expression to 0
        n_peak_labels = additional_peak_features.shape[-1]
        if n_peak_labels >= 3:
            # assuming the third column is aTPM, use aTPM to thresholding the expression
            additional_peak_features = additional_peak_features.reshape(
                -1, n_peak_labels)
            additional_peak_features[additional_peak_features[:, 2]
             < 0.1, 0] = 0
            additional_peak_features[additional_peak_features[:, 2]
             < 0.1, 1] = 0
            additional_peak_features = additional_peak_features.reshape(
                batch_size, -1, n_peak_labels)
            other_peak_labels = additional_peak_features[:, :, 2:]
            exp_label = additional_peak_features[:, :, 0:2]
        else:
            exp_label = additional_peak_features
            other_peak_labels = additional_peak_features
        exp_label = torch.from_numpy(exp_label)  # B, R, C=2 RNA+,RNA-,ATAC
        other_peak_labels = torch.from_numpy(other_peak_labels)
    else:
        exp_label = torch.FloatTensor(0)
        additional_peak_features = torch.FloatTensor(0)
        other_peak_labels = torch.FloatTensor(0)

    # return peak_signal_track, peak_sequence, sample_metadata, celltype_peaks, peak_signal_track_boundary, peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, exp_label, other_peak_labels, hic_matrix
    batch = {
        'sample_track': sample_track.float(),
        'sample_peak_sequence': sample_peak_sequence.float(),
        'metadata': metadata,
        'celltype_peaks': celltype_peaks,
        'chunk_size': chunk_size,
        'mask': mask.bool(),
        'loss_mask': get_mask_pos(mask).unsqueeze(-1).bool(),
        'padding_mask': get_padding_pos(mask).bool(),
        'n_peaks': n_peaks,
        'max_n_peaks': max_n_peaks,
        'total_peak_len': total_peak_len,
        'motif_mean_std': motif_mean_std,
        'exp_label': exp_label,
        'atpm': other_peak_labels[:, :, 0],
        'hic_matrix': hic_matrix}

    return batch


def get_perturb_collate_fn(perturbation_batch):
    # extract WT and MUT list from perturbation_batch List[dict('WT': List[dict], 'MUT': List[dict])]
    WT_batch = [p['WT'] for p in perturbation_batch]
    MUT_batch = [p['MUT'] for p in perturbation_batch]
    WT_batch = get_rev_collate_fn(WT_batch, reverse_complement=False)
    MUT_batch = get_rev_collate_fn(MUT_batch, reverse_complement=False)
    return {'WT': WT_batch, 'MUT': MUT_batch}


def everything_collate(batch):
    batch = batch_dict_list_to_dict(batch) 
    zarr_batch = get_rev_collate_fn(batch['zarr'], reverse_complement=False)
    rrd_batch = default_collate(batch['rrd'])
    # merge the two batches
    zarr_batch.update(rrd_batch)
    return zarr_batch
