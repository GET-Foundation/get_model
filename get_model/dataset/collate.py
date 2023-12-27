
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


def csr_to_torch_sparse(csr):
    # Step 1: Extract CSR components
    csr_data = csr.data
    csr_indices = csr.indices
    csr_indptr = csr.indptr

    # Step 2: Convert CSR to COO
    coo_row_indices = np.empty_like(csr_data, dtype=np.int64)
    for i in range(len(csr_indptr) - 1):
        start, end = csr_indptr[i], csr_indptr[i+1]
        coo_row_indices[start:end] = i  # Fill row indices
    coo_col_indices = csr_indices
    coo_indices = np.vstack((coo_row_indices, coo_col_indices))

    # Step 3: Create a PyTorch sparse tensor
    coo_indices = torch.LongTensor(coo_indices)
    coo_data = torch.FloatTensor(csr_data)
    size = torch.Size(csr.shape)
    return coo_indices, coo_data, size