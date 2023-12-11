import os
import os.path
from typing import Any, Callable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
# from dataset.augmentation import (
#     DataAugmentationForGETPeak,
#     DataAugmentationForGETPeakFinetune,
# )
from dataset.io import generate_paths, get_hierachical_ctcf_pos, prepare_sequence_idx
from dataset.splitter import cell_splitter, chromosome_splitter
from scipy.sparse import coo_matrix, load_npz, vstack
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm



# Two dimension to shuffle per n epoch:
# 1. Cell/Cell type based on metadata.sampling_dict: ['GM12878': (zarr_file, 0), 'K562': (zarr_file, 1), 
# 'fetal_b_cell': (zarr_file, [2,3,4,5,6,7])]
# 2. Genomic region based on CTCF/TAD boundary

# metadata will implement a .get_data(celltype_sampling) method to return 

# metadata.shuffle_sampling_dict()

# metadata.get_data(celltype_id, zarr_file, barcode_idx) -> zarr_storage

# zarr_storage.get_signal(chrom, start, end) -> signal (len(end-start))

# make_dataset(metadata, ctcf_tad_boundary)

class PretrainDataset(Dataset):

    """
    Args:
        root (string): Root directory path.
     Attributes:
        samples (list): List of (sample, cell_index) tuples
    """


# one sample for model: atac signal (N, 1000, 1), seq (N, 1000, 4) 