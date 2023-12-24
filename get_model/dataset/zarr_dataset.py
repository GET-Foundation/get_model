#%%
import os
import os.path
from posixpath import basename
from typing import Any, Callable, Optional, Tuple, List, Dict
from intervaltree import IntervalTree
import sys
from glob import glob
sys.path.append('/manitou/pmg/users/xf2217/get_model')
import numpy as np
import pandas as pd
from caesar.io.genome import ChromSize
# from augmentation import (
#     DataAugmentationForGETPeak,
#     DataAugmentationForGETPeakFinetune,
# )
from get_model.dataset.io import generate_paths, get_hierachical_ctcf_pos, prepare_sequence_idx
from get_model.dataset.splitter import cell_splitter, chromosome_splitter
from scipy.sparse import coo_matrix, load_npz, vstack
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm
from caesar.io.zarr_io import SequenceDenseZarrIO
from numcodecs import Blosc
#%%

class CelltypeDenseZarrIO(object):
    """
    Class for handling dense data in Zarr format.
    Directly use zarr index to get the data.
    """

    def __init__(self, path: str):
        self.data_key = basename(path)
        self.dataset = zarr.open_group(path, mode='r')
        self.ids = list(self.dataset.keys())
        self.celltypes = [self.dataset[id].attrs['celltype'] for id in self.ids]
        self.n_samples = [self.dataset[id].attrs['n_sample'] for id in self.ids]
        self.n_celltypes = len(self.ids)
        self.chunk_size = self.dataset[self.ids[0]]['chr1']['chunk_1'].shape[0]
        self.chrom_n_chunks = {}
        for chr_name in self.dataset[self.ids[0]].keys():
            self.chrom_n_chunks[chr_name] = len(self.dataset[self.ids[0]][chr_name])

    def get_track(self, celltype, chr_name: str, start: int, end: int) -> np.ndarray:
        """
        Get the track from the zarr file

        Args:
            celltype (str): celltype name
            chr_name (str): chromosome name
            start (int): start position
            end (int): end position

        Returns:
            np.ndarray: track
        """
        chunk_start = start // self.chunk_size
        chunk_end = end // self.chunk_size + 1
        track = []
        for chunk_idx in range(chunk_start, chunk_end):
            chunk = self.dataset[celltype][chr_name][f'chunk_{chunk_idx}']
            c_start = chunk_idx * self.chunk_size
            track.append(chunk[start - c_start:end - c_start])
        return np.concatenate(track)
    
    def sample_track(self, chr_name: str, start: int, end: int) -> np.ndarray:
        cell_type = np.random.choice(self.celltypes)
        return self.get_track(cell_type, chr_name, start, end)



#%%
class PretrainDataset(Dataset):

    """
    Args:
        root (string): Root directory path.
    """
    def __init__(self, zarr_dirs, genome_seq_zarr, insulation_path) -> None:
        from glob import glob
        super().__init__()
        self.sequence = SequenceDenseZarrIO(genome_seq_zarr+'/chrs')
        self.sequence.load_to_memory()
        self.zarr_dirs = zarr_dirs
        self.zarr_dict = {}
        self.data_keys = []
        for zarr_dir in self.zarr_dirs:
            cdz = CelltypeDenseZarrIO(zarr_dir)
            self.data_keys.append(cdz.data_key)
            self.zarr_dict[cdz.data_key] = cdz
        self.n_celltypes = np.array([z.n_celltypes for z in self.zarr_dict.values()]).sum()
        self.chunk_size = self.zarr_dict[self.data_keys[0]].chunk_size
        self.chrom_n_chunks = self.zarr_dict[self.data_keys[0]].chrom_n_chunks
        # total length of the dataset is n_chunks * n_celltypes, n_chunks is the number of chunks in each chromosome -1 (sampling 2 at a time)
        self.total_length = sum([n_chunks-1 for n_chunks in self.chrom_n_chunks.values()]) * self.n_celltypes
        self.insulation_list = self.load_insulation(insulation_path)
        self.insulation_dict = self.sample_insulation()
        return
    
    def load_insulation(self, insulation_path):
        """
        Load insulation data which is a list of pandas dataframe
        """
        for f in glob(insulation_path+'/*.bed'):
            self.insulation_list.append(pd.read_csv(f, sep='\t', header=None))
        return
    
    def sample_insulation(self):
        """
        Sample an insulation dataframe from the list for each cell type, return a dictionary mapping celltype_idx to dataframe
        """
        insulation_dict = {}
        for celltype_idx in range(self.n_celltypes):
            insulation_dict[celltype_idx] = self.insulation_list[np.random.randint(len(self.insulation_list))]
        return insulation_dict
        

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a 4Mb (2 chunks) region from a random cell type, with both atac and sequence values
        """
        # get the celltype
        celltype_idx = index // sum([n_chunks-1 for n_chunks in self.chrom_n_chunks.values()])
        celltype_insulation = self.insulation_dict[celltype_idx]
        celltype_peaks = 
        data_key, celltype = self.get_celltype(celltype_idx)
        # get the chromosome and chunk index
        chunk_idx = index % sum([n_chunks-1 for n_chunks in self.chrom_n_chunks.values()])
        chr_name, chunk_idx = self.get_chr_chunk_idx(chunk_idx)
        # get the start and end position
        start, end = self.get_start_end(chr_name, chunk_idx)
        # get the track
        track = self.zarr_dict[data_key].get_track(celltype, chr_name, start, end)
        # get the sequence
        sequence = self.sequence.get_track(chr_name, start, end)
        return track, sequence, celltype_insulation
    
    def get_celltype(self, celltype_idx):
        """
        Get the data_key and celltype from the celltype index
        """
        for data_key, zarr in self.zarr_dict.items():
            if celltype_idx < zarr.n_celltypes:
                return data_key, zarr.celltypes[celltype_idx]
            else:
                celltype_idx -= zarr.n_celltypes
        raise ValueError(f'Celltype index {celltype_idx} is out of range')

    
    def get_chr_chunk_idx(self, chunk_idx):
        """
        Get the chromosome name and chunk index from the chunk index
        """
        for chr_name, n_chunks in self.chrom_n_chunks.items():
            if chunk_idx < n_chunks-1:
                return chr_name, chunk_idx
            else:
                chunk_idx -= n_chunks-1
        raise ValueError(f'Chunk index {chunk_idx} is out of range')

    def get_start_end(self, chr_name, chunk_idx):
        """
        Get the start and end position from the chunk index
        """
        start = chunk_idx * self.chunk_size
        end = start + 2 * self.chunk_size
        return start, end
    
    def __len__(self) -> int:
        return self.total_length

#%%
import zarr
pretrain = PretrainDataset(['/pmglocal/xf2217/shendure_fetal/shendure_fetal_dense_fixed.zarr'], '/manitou/pmg/users/xf2217/get_model/data/hg38.zarr')
pretrain.chrom_n_chunks
pretrain.__len__()
# %%
for i in tqdm(range(10000)):
    try:
        pretrain.__getitem__(i)
    except:
        continue
# %%
