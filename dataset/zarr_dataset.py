import os
import os.path
from typing import Any, Callable, Optional, Tuple, List, Dict
from intervaltree import IntervalTree

import numpy as np
import pandas as pd
from caesar.io.genome import ChromSize
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

from numcodecs import Blosc

class ZarrIO:
    """
    Base class for Zarr file input/output operations.
    """

    @staticmethod
    def save_array(data: np.ndarray, path: str, dataset_name: str, chunk_shape: Tuple[int, int], compressor_options: dict) -> None:
        """
        Save a numpy array to a Zarr file with the specified dataset name and chunk shape.

        Parameters:
        data (np.ndarray): The NumPy array to save.
        path (str): The file system path for the Zarr output.
        dataset_name (str): The name of the dataset within the Zarr file.
        chunk_shape (Tuple[int, int]): The shape of the chunks in which data is stored.
        compressor_options (dict): Options for the compressor.
        """
        compressor = Blosc(**compressor_options)
        root = zarr.open_group(path, mode='a')
        z = root.zeros(name=dataset_name, shape=data.shape, chunks=chunk_shape, dtype=data.dtype, compressor=compressor)
        z[:] = data

    @staticmethod
    def load_zarr(path: str) -> Any:
        """
        Load a Zarr file from the specified path.

        Parameters:
        path (str): The file system path to the Zarr file.

        Returns:
        Any: The Zarr object loaded from the file.
        """
        return zarr.open_group(path, mode='r')

class DenseZarrIO(ZarrIO):
    """
    Class for handling dense data in Zarr format.
    Directly use zarr index to get the data.
    """

    def __init__(self, path: str):
        self.dataset = self.load_zarr(path)
        self.D = self.dataset['chrs/chr1'].shape
        if len(self.D) == 1:
            self.D = 1
        else:
            self.D = self.D[1]

    def get_track(self, chr_name: str, start: int, end: int) -> np.ndarray:
        """
        Generate a track for the specified region.

        Parameters:
        chr_name (str): The name of the chromosome.
        start (int): The start position of the region.
        end (int): The end position of the region.
        chunk_size (int): Size of the chunk.

        Returns:
        np.ndarray: The generated track for the specified region.
        """
        return self.dataset['chrs'][chr_name][start:end]

    def get_regions(self, chr_name: str, region_list: List[Tuple[int, int]]) -> np.ndarray:
        """
        Generate a region set for the specified regions.

        Parameters:
        chr_name (str): The name of the chromosome.
        region_list (List[Tuple[int, int]]): The list of regions to generate the region set for.

        Returns:
        np.ndarray: The generated region set for the specified regions.
        """
        if self.D == 1:
            region_set = np.zeros((len(region_list), region_list[0][1] - region_list[0][0]))
        else:
            region_set = np.zeros((len(region_list), region_list[0][1] - region_list[0][0], self.D))
        a = self.get_track(chr_name, region_list[0][0], region_list[-1][1]) # N, L, D
        for i, (start, end) in enumerate(region_list):
            region_set[i] = a[start:end]
        return region_set # N, L, D

class PretrainDataset(Dataset):

    """
    Args:
        root (string): Root directory path.
     Attributes:
        metadata (pandas.DataFrame): Metadata of the dataset, for each sample annotate their libsize, celltype, source, etc..
        samples (dict): Dict of (sample_id: atac_zarr_file, peak_bed, assembly) tuples
        sequence_dir: directory with zarr file for genome sequences
    """
    def __init__(self, zarr_dirs, ctcf_df) -> None:
        super().__init__()
        self.zarr_dirs = zarr_dirs
        for zarr_dir in self.zarr_dirs:
            assert os.path.exists(zarr_dir), f"zarr_dir {zarr_dir} does not exist"
        

# /manitou/pmg/users/xf2217/Seq2VecDB_ENCODE/assays/atac_seq/targets/atac/cell_types/IMR_90/conditions/Homo_sapiens_IMR_90_nuclear_fraction_and_unspecified_fraction/dataset_accession_numbers/ENCSR200OML/reps_subsampled/40000000/genome_assembly/hg38/steps/S07_zarr/zarrs_normalized/data.zarr/



    # one sample for model: atac signal (N, 1000, 1), seq (N, 1000, 4) 
        
#%%

#%%
