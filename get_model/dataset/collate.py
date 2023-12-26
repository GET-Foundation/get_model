
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


