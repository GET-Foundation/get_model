# %%
import logging
from operator import is_
import os.path
import random
import re
import sys
import time
import warnings
from glob import glob
from math import log
from posixpath import basename
from queue import Queue
import torch

import numpy as np
import pandas as pd
import zarr
from caesar.io.zarr_io import CelltypeDenseZarrIO, DenseZarrIO
from pyranges import PyRanges as pr
from scipy.sparse import csr_matrix, vstack
from torch.utils.data import Dataset
from tqdm import tqdm

from get_model.dataset.io import (generate_paths, get_hierachical_ctcf_pos,
                                  prepare_sequence_idx)
from get_model.dataset.splitter import cell_splitter, chromosome_splitter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Suppress all deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append('/manitou/pmg/users/xf2217/get_model')

           
class MotifMeanStd(object):
    """A class that reads the mean and std of motif scores from a zarr file.
    e.g. z['mean_std/chr1']
    """
    def __init__(self, zarr_path):
        import glob
        self.zarr_path = zarr_path
        self.zarr = zarr.open(zarr_path, mode='r')
        self.chromosomes = [basename(path) for path in glob.glob(os.path.join(zarr_path, 'mean_std/*'))]
        self.data_dict = {chromosome: self.zarr['mean_std/'+chromosome][:] for chromosome in self.chromosomes}

class ZarrDataPool(object):
    """A class to handle data loading for a slot."""
    def __init__(self, zarr_dirs, genome_seq_zarr, insulation_paths, peak_name='peaks',insulation_subsample_ratio=0.1,
                 max_peak_length=None, center_expand_target=None, sequence_obj=None, motif_mean_std_obj=None,
                 additional_peak_columns=None,
                 leave_out_celltypes=None, leave_out_chromosomes=None, is_train=True):
        logging.info('Initializing ZarrDataPool')
        if sequence_obj is None:
            self.sequence = DenseZarrIO(genome_seq_zarr, dtype='int8', mode='r')
            self.sequence.load_to_memory_dense()
        else:
            self.sequence = sequence_obj
        self.motif_mean_std_obj = motif_mean_std_obj
        self.zarr_dirs = zarr_dirs
        self.insulation_paths = insulation_paths
        self.insulation_subsample_ratio = insulation_subsample_ratio
        self.peak_name = peak_name
        self.max_peak_length = max_peak_length
        self.center_expand_target = center_expand_target
        self.leave_out_celltypes = leave_out_celltypes
        self.leave_out_chromosomes = leave_out_chromosomes
        self.additional_peak_columns = additional_peak_columns
        self.is_train = is_train
        self.initialize_datasets()
        self.calculate_metadata()
        logging.info('ZarrDataPool initialized')
    
    def initialize_datasets(self):
        self.zarr_dict = {cdz.data_key: cdz for zarr_dir in self.zarr_dirs for cdz in [
            CelltypeDenseZarrIO(zarr_dir).subset_celltypes_with_data_name(self.peak_name)]}
        
        # remove the leave out celltypes using subset
        if self.leave_out_celltypes is not None and isinstance(self.leave_out_celltypes, list):
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update({data_key: cdz.leave_out_celltypes(self.leave_out_celltypes, inverse = not self.is_train)})
        elif isinstance(self.leave_out_celltypes, str):
            # remove the leave out celltypes using substring
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update({data_key: cdz.leave_out_celltypes_with_pattern(self.leave_out_celltypes, inverse = not self.is_train)})

        self.data_keys = list(self.zarr_dict.keys())
        self.peaks_dict = self._load_peaks()
        self.insulation = self._load_insulation(
            self.insulation_paths).sample(frac=self.insulation_subsample_ratio).reset_index(drop=True)
        # remove the leave out chromosomes from insulation
        if self.leave_out_chromosomes is not None and isinstance(self.leave_out_chromosomes, list):
            if self.is_train:
                self.insulation = self.insulation.query('Chromosome not in @self.leave_out_chromosomes').reset_index(drop=True)
            else:
                self.insulation = self.insulation.query('Chromosome in @self.leave_out_chromosomes').reset_index(drop=True)
        
    def calculate_metadata(self):
        first_zarr = next(iter(self.zarr_dict.values()))
        self.n_celltypes = sum(
            zarr.n_celltypes for zarr in self.zarr_dict.values())
        self.chunk_size = first_zarr.chunk_size
        self.chrom_n_chunks = first_zarr.chrom_n_chunks
        self.genome_chunk_length = sum(
            n_chunks - 1 for n_chunks in self.chrom_n_chunks.values())
        self.total_chunk_length = self.genome_chunk_length * self.n_celltypes
    
    def load_window_data(self, window_index):
        data_key, celltype_id = self._get_celltype_info(window_index)
        chr_name, chunk_idx, start, end = self._get_chromosome_info(
            window_index)

        celltype_peaks = self._query_peaks(celltype_id, chr_name, start, end)
        item_insulation = self._query_insulation(chr_name, start, end)

        track = self.zarr_dict[data_key].get_track(
            celltype_id, chr_name, start, end, sparse=True).T.astype(np.uint16)
        # sequence = self.sequence.get_track(chr_name, start, end, sparse=False)

        # peak_sequence = self._generate_peak_sequence(
            # celltype_peaks, sequence, start, end)
        item_insulation = item_insulation.reset_index(drop=True).reset_index()
        item_insulation['key'] = str(
            window_index) + '_' + item_insulation['index'].astype(str)
        celltype_peaks = celltype_peaks.reset_index(drop=True).reset_index()
        if self.motif_mean_std_obj is not None:
            motif_mean_std = self.motif_mean_std_obj.data_dict[chr_name][chunk_idx:chunk_idx+2].reshape(2, 2, -1).mean(0)
        else:
            motif_mean_std = np.zeros((2, 1274))
        return window_index, chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std
    
    def _get_peak_names(self, data_key, celltype_id):
        """Return a list of peak names for a celltype, use glob peaks*"""
        return [key for key in self.zarr_dict[data_key].dataset[celltype_id].keys() if 'peaks' in key]

    def _load_peaks(self):
        """
        Load peaks data which is a dictionary of pandas dataframe feather
        """
        logging.info('Loading peaks data')
        peaks_dict = {}
        for data_key, cdz in self.zarr_dict.items():
            for celltype_id in cdz.ids:
                # check if the peak name exists in the zarr
                if self.peak_name not in self._get_peak_names(data_key, celltype_id):
                    continue
                peak = cdz.get_peaks(
                    celltype_id, self.peak_name)
                if self.max_peak_length is not None:
                    peak = peak.query(
                        'End-Start<@self.max_peak_length').reset_index(drop=True)
                if self.center_expand_target is not None:
                    # peak_shorter = peak.query(
                        # 'End-Start<@self.center_expand_target').reset_index(drop=True)
                    # peak_longer = peak.query(
                        # 'End-Start>=@self.center_expand_target').reset_index(drop=True)
                    peak['Start'] = (peak['Start'] + peak['End']) // 2 - \
                        self.center_expand_target // 2
                    peak['End'] = peak['Start'] + \
                        self.center_expand_target
                    # peak = pd.concat([peak_shorter, peak_longer]).sort_values(
                        # ['Chromosome', 'Start']).reset_index(drop=True)
                peaks_dict[celltype_id] = peak
        

        return peaks_dict

    def _load_insulation(self, insulation_paths: list):
        """
        Load insulation data which is a list of pandas dataframe feather
        """
        logging.info('Loading insulation data')
        insulation_list = []
        for path in insulation_paths:
            insulation_list.append(pd.read_feather(path))

        return pd.concat(insulation_list).drop_duplicates(subset=['Chromosome', 'Start', 'End'])


    def _get_celltype_info(self, window_index):
        celltype_idx = window_index // self.genome_chunk_length
        data_key, celltype_id = self._get_celltype(celltype_idx)
        return data_key, celltype_id

    def _get_chromosome_info(self, window_index):
        chunk_idx = window_index % self.genome_chunk_length
        chr_name, chunk_idx = self._get_chr_chunk_idx(chunk_idx)
        start, end = self._get_start_end(chr_name, chunk_idx)
        return chr_name, chunk_idx, start, end

    def _generate_peak_sequence(self, celltype_peaks, sequence, window_start, window_end):
        peak_sequence = np.zeros_like(sequence, dtype=np.int8)
        for start, end in celltype_peaks: #[['Start', 'End']].to_numpy()
            peak_sequence[start-window_start:end-window_start] = 1
        return csr_matrix(peak_sequence*sequence)

    def _query_peaks(self, celltype_id, chr_name, start, end):
        return self.peaks_dict[celltype_id].query(
            'Chromosome == @chr_name and Start >= @start and End <= @end')

    def _query_insulation(self, chr_name, start, end):
        return self.insulation.query(
            'Chromosome == @chr_name and Start >= @start and End <= @end')
    
    def _get_celltype(self, celltype_idx):
        """
        Get the data_key and celltype from the celltype index
        """
        for data_key, cdz in self.zarr_dict.items():
            if celltype_idx < cdz.n_celltypes:
                return data_key, cdz.ids[celltype_idx]
            else:
                celltype_idx -= cdz.n_celltypes
        raise ValueError(f'Celltype index {celltype_idx} is out of range')

    def _get_chr_chunk_idx(self, chunk_idx):
        """
        Get the chromosome name and chunk index from the chunk index
        """
        for chr_name, n_chunks in self.chrom_n_chunks.items():
            if chunk_idx < n_chunks-1:
                return chr_name, chunk_idx
            else:
                chunk_idx -= n_chunks-1
        raise ValueError(f'Chunk index {chunk_idx} is out of range')

    def _get_start_end(self, chr_name, chunk_idx):
        """
        Get the start and end position from the chunk index
        """
        start = chunk_idx * self.chunk_size
        end = start + 2 * self.chunk_size
        return start, end

class PreloadDataPack(object):
    """A class to store preloaded data for a slot."""
    def __init__(self, preload_count: int, zarr_data_pool: ZarrDataPool, padding=50, mask_ratio=0.5,
                 n_peaks_lower_bound=5, n_peaks_upper_bound=200, window_index=None):
        logging.info('Initializing PreloadDataPack')
        self.preload_count = preload_count
        self.zarr_data_pool = zarr_data_pool
        self.preloaded_data = []
        self.insulation_peak_counts = pd.DataFrame()
        self.preloaded_data_window_indices_mapping = {}
        self.next_sample = 0
        self.padding = padding
        self.mask_ratio = mask_ratio
        self.n_peaks_lower_bound = n_peaks_lower_bound
        self.n_peaks_upper_bound = n_peaks_upper_bound
        self.additional_peak_columns = self.zarr_data_pool.additional_peak_columns
        if window_index is None:
            window_index = np.random.randint(0, self.zarr_data_pool.total_chunk_length, size=self.preload_count)
        self.window_index = window_index
        self.preload_data(window_index)

        logging.info('PreloadDataPack initialized')

    def __len__(self):
        return self.insulation_peak_counts.shape[0]
    
    # def __iter__(self):
    #     return self
    
    def get_next_sample(self):
        if self.next_sample >= self.insulation_peak_counts.shape[0]:
            return None
        else:
            sample = self.get_sample_with_idx(self.next_sample)
            self.next_sample += 1
            return sample
    
    def preload_data(self, window_index=None):
        # Preload data
        # Load data for a fixed number of windows
        self.preloaded_data_window_indices_mapping = {}
        for i, window_index in enumerate(window_index):
            self.preloaded_data.append(self.zarr_data_pool.load_window_data(window_index))
            self.preloaded_data_window_indices_mapping[window_index] = i
        # trigger the computation of peak_num_per_sample
        self.insulation_peak_counts = self._calculate_peak_num_per_sample()
        if self.insulation_peak_counts.shape[0] == 0:
            logging.info('No valid insulation peak count')
            return PreloadDataPack(self.preload_count, self.zarr_data_pool, self.padding, self.mask_ratio, self.n_peaks_lower_bound, self.n_peaks_upper_bound)
        

    def get_sample_with_idx(self, idx):
        return self._get_sample_with_key(self.insulation_peak_counts.iloc[idx]['key'])

    def _get_sample_with_key(self, key):
        window_index = int(key.split('_')[0])
        insulation_index = int(key.split('_')[1])
        window_slot = self.preloaded_data_window_indices_mapping[window_index]
        return self._extract_sample_from_window(self.preloaded_data[window_slot], insulation_index)

    def _extract_sample_from_window(self, window, insulation_index):
        """
        Extract a single sample from a preloaded window.

        This method selects a portion of the data within a preloaded 4Mbp window based on insulation data.
        It extracts a sample track and peak sequence from the window for further processing.

        Parameters:
        window (tuple): Contains preloaded data for a window, including window index, chromosome name,
                        start and end positions, cell type ID, track data, peak sequence, and insulation data.

        Returns:
        tuple: A tuple containing the extracted sample track, peak sequence, and a dictionary with metadata
            about the extracted sample, including cell type ID, chromosome name, and positions.

        Note:
        This implementation relies on the availability of insulation data. If insulation data is empty,
        it attempts to load data from another randomly selected window.
        """
        # window_index, chr_name, start, end, celltype_id, track, peak_sequence, insulations, celltype_peaks = window
        window_index, chr_name, start, end, celltype_id, track, insulations, celltype_peaks, motif_mean_std = window
        if len(insulations) == 0:
            raise ValueError('Empty insulation')
        i_start, i_end = self._insulation_sampler(insulations, insulation_index)
        celltype_peaks = celltype_peaks.query('Start>@i_start and End<@i_end')
        
        if self.additional_peak_columns is not None:
            # assume numeric columns
            additional_peak_columns_data = celltype_peaks[self.additional_peak_columns].to_numpy().astype(np.float32)
        else:
            additional_peak_columns_data = None
        
        celltype_peaks = celltype_peaks[['Start', 'End']].to_numpy().astype(np.int64)
        if len(celltype_peaks) == 0:
            raise ValueError('No peaks in insulation region')

        sequence = self.zarr_data_pool.sequence.get_track(
            chr_name, i_start, i_end, sparse=False)
        sample_peak_sequence = self.zarr_data_pool._generate_peak_sequence(
            celltype_peaks, sequence, i_start, i_end)

        wi_start, wi_end = i_start - start, i_end - start
        celltype_peaks = celltype_peaks - i_start
        # TODO: padding might lead to out of bound of window

        
        sample_track = track[wi_start:wi_end]

        sample_peak_sequence = vstack(
            [sample_peak_sequence[start-self.padding:end+self.padding] for start, end in celltype_peaks])
        sample_track = vstack(
            [sample_track[start-self.padding:end+self.padding] for start, end in celltype_peaks])
        sample_metadata = {
            'celltype_id': celltype_id, 'chr_name': chr_name,
            'start': start, 'end': end, 'i_start': wi_start, 'i_end': wi_end, 'mask_ratio': self.mask_ratio
        }

        return sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, motif_mean_std, additional_peak_columns_data
    
    def _insulation_sampler(self, insulation_df, insulation_index=None):
        """
        Sample insulation from the insulation dataframe
        """
        if insulation_index is None:
            insulation_index = np.random.randint(0, len(insulation_df))
        i_start, i_end = insulation_df.iloc[insulation_index][['Start', 'End']]
        return i_start, i_end


    def _calculate_peak_num_per_sample(self):
        insulation_pool_peak_num = {}
        for window_index, chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std in self.preloaded_data:
            try:
                insulation_pool_peak_num.update(
                    self._get_peak_count(item_insulation, celltype_peaks))
            except:
                continue
        # Convert to DataFrame and sort
        insulation_pool_peak_num_df = pd.DataFrame.from_dict(
            insulation_pool_peak_num, orient='index').reset_index()
        if insulation_pool_peak_num_df.shape[0] == 0:
            return pd.DataFrame(columns = ['key', 'peak_num'])
        insulation_pool_peak_num_df.columns = ['key', 'peak_num']
        # insulation_pool_peak_num_df.sort_values('peak_num', inplace=True)
        # shuffle the dataframe
        insulation_pool_peak_num_df = insulation_pool_peak_num_df.query('peak_num>=@self.n_peaks_lower_bound and peak_num<=@self.n_peaks_upper_bound')
        insulation_pool_peak_num_df = insulation_pool_peak_num_df.sample(frac=1).reset_index(drop=True)
        return insulation_pool_peak_num_df

    def _get_peak_count(self, item_insulation, celltype_peaks):
        df = pr(item_insulation).join(pr(celltype_peaks), suffix="_peak").df.groupby(
            'key').index_peak.count().reset_index()
        return df.set_index('key').to_dict()['index_peak']

class PretrainDataset(Dataset):
    def __init__(self, zarr_dirs, genome_seq_zarr, genome_motif_zarr, insulation_paths, peak_name='peaks', additional_peak_columns=None, preload_count=50, padding=50, mask_ratio=0.5, n_packs=2, max_peak_length=None, center_expand_target=None, n_peaks_lower_bound=5, n_peaks_upper_bound=200, sequence_obj=None,
                leave_out_celltypes=None, leave_out_chromosomes=None, is_train=True, dataset_size=655_360):
        super().__init__()
        """
        Pretrain dataset for GET model.

        This dataset is used to train the GET model in a self-supervised manner.
        It loads data from a set of zarr files and generates samples for training.
        Each sample consists of a track and a peak sequence extracted from a 4Mbp window.
        The track is a 2D sparse array of shape (nucleotide,1) and the peak sequence is a 2D array of shape (nucleotide,4). Peak sequence is generated by multiplying the peak mask with the genomic sequence. Track is cell-type specific ATAC-seq insertion counts per nucleotide, without any normalization. Peak sequence is a one-hot encoding of DNA. Other metadata about the sample is also included in the sample as a dictionary.

        Parameters:
        zarr_dirs (list): A list of paths to zarr files.
        genome_seq_zarr (str): Path to the genome sequence zarr file.
        insulation_paths (list): A list of paths to insulation data.
        preload_count (int): Number of windows to preload.
        samples_per_window (int): Number of samples to generate from each window.

        Attributes:
        zarr_dirs (list): A list of paths to zarr files.
        genome_seq_zarr (str): Path to the genome sequence zarr file.
        insulation_paths (list): A list of paths to insulation data.
        preload_count (int): Number of windows to preload.
        samples_per_window (int): Number of samples to generate from each window.
        sequence (DenseZarrIO): An instance of DenseZarrIO for genome sequence.
        zarr_dict (dict): A dictionary of CelltypeDenseZarrIO instances for zarr files.
        data_keys (list): A list of data keys for zarr files.
        n_celltypes (int): Number of cell types.
        chunk_size (int): Chunk size.
        chrom_n_chunks (dict): A dictionary of chromosome names and number of chunks.
        genome_chunk_length (int): Total number of chunks in the genome.
        total_chunk_length (int): Total number of chunks in the genome multiplied by number of cell types.
        preloaded_data (list): A list of preloaded data for windows.
        usage_counters (list): A list of usage counters for preloaded data.
        locks (list): A list of locks for preloaded data.
        reload_queue (Queue): A queue for reloading data.
        reload_thread (threading.Thread): A thread for reloading data.
        peaks_dict (dict): A dictionary of pandas dataframes for peaks data.
        insulation (pd.DataFrame): A pandas dataframe for insulation data.

        Note:
        This implementation features a preloading mechanism to speed up data loading.
        It preloads a fixed number of windows and generates samples from the preloaded data.
        When a window is used up, it reloads the data for the window in a separate thread.
        Note that a window is 4Mbp in size while a chunk in zarr is 2Mbp in size, so each window contains two consecutive chunks.

        Returns:
        tuple: A tuple containing the extracted sample track, peak sequence, and a dictionary with metadata
            about the extracted sample, including cell type ID, chromosome name, and positions.
        """
        self.preload_count = preload_count
        self.padding = padding
        self.mask_ratio = mask_ratio

        self.peak_name = peak_name
        self.n_peaks_lower_bound = n_peaks_lower_bound
        self.n_peaks_upper_bound = n_peaks_upper_bound

        self.leave_out_celltypes = leave_out_celltypes
        self.leave_out_chromosomes = leave_out_chromosomes
        self.is_train = is_train
        self.dataset_size = dataset_size
        self.n_packs = n_packs
        self.additional_peak_columns = additional_peak_columns
        if sequence_obj is None:
            self.sequence = DenseZarrIO(genome_seq_zarr, dtype='int8', mode='r')
            self.sequence.load_to_memory_dense()
        else:
            self.sequence = sequence_obj
        self.mms = MotifMeanStd(genome_motif_zarr)
        self.datapool = ZarrDataPool(zarr_dirs, genome_seq_zarr, insulation_paths, peak_name=peak_name, max_peak_length=max_peak_length, center_expand_target=center_expand_target, sequence_obj=self.sequence,
                                     motif_mean_std_obj=self.mms,
                                     additional_peak_columns=self.additional_peak_columns,
                                     leave_out_celltypes=self.leave_out_celltypes, 
                                     leave_out_chromosomes=self.leave_out_chromosomes,
                                     is_train=self.is_train, )

        # initialize n_packs preload data packs
        self.preload_data_packs = None
        # self.locks = [threading.Lock() for _ in range(n_packs)]
        self.current_pack = 0
        self.avaliable_packs = list(range(n_packs))

    def __getitem__(self, index: int):
        return self._getitem(index)

    def _getitem(self, index: int):
        """
        Load item from current preload data pack, after
        the current pack is used up, switch to the next avaliable pack and reload the current pack
        """
        sample = self.preload_data_packs[self.current_pack].get_next_sample()

        if sample is None:
            # remove the current pack from avaliable packs
            self.avaliable_packs.remove(self.current_pack)
            #  reload the current pack
            self.reload_data(self.current_pack)
            # switch to the next avaliable pack
            self.current_pack = (self.current_pack+1) % self.n_packs
            return self._getitem(index)
        else:
            return sample
            

    def __len__(self):
        # Return the length of the dataset
        # Implement based on how you define the length of your dataset
        # Could be based on total windows, number of samples per window, etc.
        return self.dataset_size


    def reload_data(self, slot):
        # This method runs in a separate thread
        # logging.info(f'Async reloading data for slot {index}')
        # reload by reinitializing the preload data pack and put it back to the preload_data_packs
        self.preload_data_packs[slot] = PreloadDataPack(
            self.preload_count, self.datapool, self.padding, self.mask_ratio, self.n_peaks_lower_bound, self.n_peaks_upper_bound)
        # self.preload_data_packs[slot].preload_data()
        # add the index back to avaliable packs
        self.avaliable_packs.append(slot)


def worker_init_fn_get(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(torch.initial_seed() + worker_id)
    
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.preload_data_packs is None:
        dataset.preload_data_packs = [PreloadDataPack(dataset.preload_count, dataset.datapool, dataset.padding, dataset.mask_ratio,
        dataset.n_peaks_lower_bound, dataset.n_peaks_upper_bound) for _ in range(dataset.n_packs)]