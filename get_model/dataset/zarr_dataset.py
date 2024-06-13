import logging
import os
import os.path
import sys
import warnings
from dataclasses import dataclass
from posixpath import basename
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import zarr
from caesar.io.gencode import Gencode
from caesar.io.zarr_io import CelltypeDenseZarrIO, DenseZarrIO
from pyranges import PyRanges as pr
from scipy.sparse import coo_matrix, csr_matrix, load_npz, vstack
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Suppress all deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _chromosome_splitter(all_chromosomes: list, leave_out_chromosomes: str | None, is_train=True):
    input_chromosomes = all_chromosomes.copy()
    if leave_out_chromosomes is None:
        leave_out_chromosomes = []
    elif ',' in leave_out_chromosomes:
        leave_out_chromosomes = leave_out_chromosomes.split(",")
    else:
        leave_out_chromosomes = [leave_out_chromosomes]

    if is_train or leave_out_chromosomes == [""] or leave_out_chromosomes == []:
        input_chromosomes = [
            chrom for chrom in input_chromosomes if chrom not in leave_out_chromosomes]
    else:
        input_chromosomes = all_chromosomes if leave_out_chromosomes == [
        ] else leave_out_chromosomes

    if isinstance(input_chromosomes, str):
        input_chromosomes = [input_chromosomes]

    return input_chromosomes


def get_padding_pos(mask):
    mask_ = mask.clone()
    mask_[mask_ != -10000] = 0
    mask_[mask_ != 0] = 1
    return mask_


def get_mask_pos(mask):
    mask_ = mask.clone()
    mask_[mask_ == -10000] = 0
    return mask_


def get_sequence_with_mutations(arr, start, end, mut):
    from atac_rna_data_processing.io.sequence import DNASequence
    arr_wt = arr.copy()
    mut_df_chr = mut.query('Start>=@start & End <= @end').sort_values('Start')

    offset = 0  # Track the net offset introduced by mutations

    for _, row in mut_df_chr.iterrows():
        # Adjust mutation positions by the current offset
        mut_start = row.Start - start + offset
        mut_end = row.End - start + offset

        # Convert reference and alternative alleles to one-hot encoded format
        wt_on_genome = arr[mut_start:mut_end]
        wt_in_mut_file = DNASequence(row.Ref).one_hot.astype(float)
        alt_sequence = DNASequence(row.Alt).one_hot.astype(float)

        # Ensure the sequence in the genome matches the reference sequence
        if not np.array_equal(wt_on_genome, wt_in_mut_file):
            logging.warning(
                f"Reference sequence in mutation file does not match the genome sequence at {row.Start}-{row.End}. Skipping this mutation.")
            continue

        # Apply the mutation and calculate the new offset
        prev_length = mut_end - mut_start
        new_length = len(alt_sequence)
        if new_length != prev_length:
            arr = apply_indel(arr, mut_start, mut_end, alt_sequence)
            offset += new_length - prev_length
        else:
            arr[mut_start:mut_end] = alt_sequence
        # Update the offset for subsequent mutations

    return arr, arr_wt


def _stack_tracks_with_padding_and_inactivation(celltype_peaks, track, padding, inactivated_peak_idx):
    # Initialize an empty list to hold the arrays.
    stacked_track_list = []
    track_shape = track.shape
    track_dtype = track.dtype
    if len(track_shape) == 1:
        track_depth = 1
    else:
        track_depth = track_shape[1]
    # Iterate over the enumerated celltype_peaks to apply padding and handle inactivated peaks.
    if inactivated_peak_idx is None:
        inactivated_peak_idx = []
    for i, (start, end) in enumerate(celltype_peaks):
        if i not in inactivated_peak_idx:
            # Apply padding and add the sliced track to the list.
            padded_track = track[max(0, start-padding):end+padding]
            # check if the padded track is empty
            if padded_track.size == 0 or padded_track.shape[0] <= 1:
                padded_track = csr_matrix(
                    (end-start+2*padding, track_depth), dtype=track_dtype)
            stacked_track_list.append(padded_track)
        else:
            # Create a zero array of the required shape for inactivated peaks.
            zero_array = csr_matrix(
                (end-start+2*padding, track_depth), dtype=track_dtype)
            stacked_track_list.append(zero_array)

    # Vertically stack the arrays.
    stacked_track = vstack(stacked_track_list)
    return stacked_track


def apply_indel(arr, start, end, alt_sequence):
    """Helper function to apply a mutation to the array."""
    # Remove the original sequence
    arr = np.delete(arr, slice(start, end), axis=0)
    # Insert the new sequence
    arr = np.insert(arr, start, alt_sequence, axis=0)
    return arr


def get_hic_from_idx(hic, csv, start=None, end=None, resolution=5000, method='observed'):
    # if from hic straw
    if hasattr(hic, 'getMatrixZoomData'):
        if start is not None and end is not None:
            csv_region = csv.iloc[start:end]
        else:
            csv_region = csv
        chrom = csv_region.iloc[0].Chromosome.replace("chr", "")
        if chrom != csv_region.iloc[-1].Chromosome.replace("chr", ""):
            return None
        start = csv_region.iloc[0].Start // resolution
        end = csv_region.iloc[-1].End // resolution + 1
        if (end-start) * resolution > 4000000:
            return None
        hic_idx = np.array([row.Start // resolution - start +
                        1 for _, row in csv_region.iterrows()])
        mzd = hic.getMatrixZoomData('chr' + chrom, 'chr' + chrom, method, "SCALE", "BP", resolution)
        numpy_matrix = mzd.getRecordsAsMatrix(
            start * resolution, end * resolution, start * resolution, end * resolution)
        numpy_matrix = np.nan_to_num(numpy_matrix)
        dst = np.log10(numpy_matrix[hic_idx, :][:, hic_idx]+1)
        return dst
    # if from cooler
    elif hasattr(hic, 'matrix'):
        if start is not None and end is not None:
            csv_region = csv.iloc[start:end]
        else:
            csv_region = csv
        chrom = csv_region.iloc[0].Chromosome.replace("chr", "")
        if chrom != csv_region.iloc[-1].Chromosome.replace("chr", ""):
            return None
        start = csv_region.iloc[0].Start // resolution
        end = csv_region.iloc[-1].End // resolution + 1
        if (end-start) * resolution > 4000000:
            return None
        hic_idx = np.array([row.Start // resolution - start for _, row in csv_region.iterrows()])
        numpy_matrix = hic.matrix(balance=True).fetch(f'chr{chrom}:{start * resolution}-{end * resolution}')
        numpy_matrix = np.nan_to_num(numpy_matrix)
        dst = np.log10(numpy_matrix[hic_idx, :][:, hic_idx]+1)
        return dst


class MotifMeanStd(object):
    """A class that reads the mean and std of motif scores from a zarr file.
    e.g. z['mean_std/chr1']
    """

    def __init__(self, zarr_path):
        import glob
        self.zarr_path = zarr_path
        self.zarr = zarr.open(zarr_path, mode='r')
        self.chromosomes = [basename(path) for path in glob.glob(
            os.path.join(zarr_path, 'mean_std/*'))]
        self.data_dict = {
            chromosome: self.zarr['mean_std/'+chromosome][:] for chromosome in self.chromosomes}


def get_sequence_obj(genome_seq_zarr: dict | str):
    """
    Get DenseZarrIO object for genome sequence.
    """
    if isinstance(genome_seq_zarr, dict):
        sequence_obj = {}
        for assembly, zarr_file in genome_seq_zarr.items():
            sequence_obj[assembly] = DenseZarrIO(
                zarr_file, dtype='int8', mode='r')
            sequence_obj[assembly].load_to_memory_dense()
    elif isinstance(genome_seq_zarr, str):
        assembly = basename(genome_seq_zarr).split('.')[0]
        sequence_obj = {assembly: DenseZarrIO(
            genome_seq_zarr, dtype='int8', mode='r')}
        sequence_obj[assembly].load_to_memory_dense()
    return sequence_obj


def get_gencode_obj(genome_seq_zarr: dict | str, gtf_dir: str = '.'):
    """
    Get Gencode object for genome sequence.
    """
    if isinstance(genome_seq_zarr, dict):
        gencode_obj = {}
        for assembly, _ in genome_seq_zarr.items():
            gencode_obj[assembly] = Gencode(assembly, gtf_dir=gtf_dir)
    elif isinstance(genome_seq_zarr, str):
        assembly = basename(genome_seq_zarr).split('.')[0]
        gencode_obj = {assembly: Gencode(assembly, gtf_dir=gtf_dir)}
    return gencode_obj


class ZarrDataPool(object):
    """This class loads data from a set of zarr files and generates samples for training.

    Each sample consists of a track and a peak sequence extracted from a 4Mbp window.

    The track is a 2D sparse array of shape (nucleotide,1) and the peak sequence is a 2D array of shape (nucleotide,4). Peak sequence is generated by multiplying the peak mask with the genomic sequence. Track is cell-type specific ATAC-seq insertion counts per nucleotide, without any normalization. Peak sequence is a one-hot encoding of DNA. Other metadata about the sample is also included in the sample as a dictionary.

    Parameters:
    zarr_dirs (list): A list of paths to zarr files.
    genome_seq_zarr (str): Path to the genome sequence zarr file.
    insulation_paths (list): A list of paths to insulation data.
    peak_name (str): The name of the peak track in the zarr files. Default is 'peaks'.
    negative_peak_name (str): The name of the negative peak track in the zarr files. Default is None.
    negative_peak_ratio (float): The ratio of negative peaks to include in the dataset. Default is 0.1.
    insulation_subsample_ratio (float): The ratio of insulation data to use. Default is 0.1.
    max_peak_length (int): The maximum length of peaks to include in the dataset. Default is None.
    center_expand_target (int): The target length of peaks to center and expand. Default is None.
    sequence_obj (DenseZarrIO): A DenseZarrIO object for genome sequence. Default is None.
    motif_mean_std_obj (MotifMeanStd): A MotifMeanStd object for motif mean and std. Default is None.
    additional_peak_columns (list): A list of additional peak columns to include in the dataset. Default is None.
    leave_out_celltypes (list): A list of cell type IDs to leave out. Default is None.
    leave_out_chromosomes (list): A list of chromosome names to leave out. Default is None.
    non_redundant (str): Whether to remove redundant cell type instances. Default is 'max_depth'.
    random_shift_peak (bool): Whether to randomly shift peaks. Default is None.
    filter_by_min_depth (bool): Whether to filter out samples by minimum depth. Default is None.
    is_train (bool): Whether to use the dataset for training. Default is True.
    hic_path (str): Path to the HiC data file. Default is None.

    Returns:
    ZarrDataPool: A ZarrDataPool object.

    Note:
    This class is used by PretrainDataset to load data from zarr files.
    """

    def __init__(self, zarr_dirs=None, genome_seq_zarr=None, sequence_obj=None, insulation_paths=None, peak_name='peaks', negative_peak_name=None, negative_peak_ratio=0.1, insulation_subsample_ratio=0.1,
                 max_peak_length=None, center_expand_target=None,
                 motif_mean_std_obj=None, additional_peak_columns=None, keep_celltypes=None, leave_out_celltypes=None,
                 leave_out_chromosomes=None, peak_count_filter=0, non_redundant='max_depth', random_shift_peak=None,
                 filter_by_min_depth=None, is_train=True, hic_path=None):
        # Rest of the code
        logging.info('Initializing ZarrDataPool')
        self.sequence = sequence_obj if sequence_obj is not None else get_sequence_obj(
            genome_seq_zarr)
        self.motif_mean_std_obj = motif_mean_std_obj
        self.zarr_dirs = zarr_dirs
        self.insulation_paths = insulation_paths
        self.insulation_subsample_ratio = insulation_subsample_ratio
        self.peak_name = peak_name
        self.negative_peak_name = negative_peak_name
        self.max_peak_length = max_peak_length
        self.center_expand_target = center_expand_target
        self.keep_celltypes = keep_celltypes
        self.leave_out_celltypes = leave_out_celltypes
        self.leave_out_chromosomes = leave_out_chromosomes
        self.peak_count_filter = peak_count_filter
        self.additional_peak_columns = additional_peak_columns
        self.is_train = is_train
        self.negative_peak_ratio = negative_peak_ratio
        self.random_shift_peak = random_shift_peak
        self.non_redundant = non_redundant
        self.filter_by_min_depth = filter_by_min_depth
        self.hic_path = hic_path
        self.hic_obj = None
        self.initialize_datasets()
        self.calculate_metadata()
        self.celltype_to_data_key
        logging.info('ZarrDataPool initialized')

    @property
    def celltype_to_data_key(self):
        if not hasattr(self, '_celltype_to_data_key'):
            celltype_to_data_key = {}
            for data_key, cdz in self.zarr_dict.items():
                for celltype_id in cdz.ids:
                    celltype_to_data_key[celltype_id] = data_key
            self._celltype_to_data_key = celltype_to_data_key
        return self._celltype_to_data_key

    def initialize_datasets(self):
        """
        Initialize the zarr datasets and load peaks and insulation data.
        """
        self._subset_datasets()
        self.data_keys = list(self.zarr_dict.keys())
        self.assembly_dict = {data_key: cdz.assembly for data_key,
                              cdz in self.zarr_dict.items()}
        self.peaks_dict = self._load_peaks(
            self.peak_name, self.peak_count_filter)
        self.insulation = self._load_insulation()
        self.hic_obj = self._load_hic()

    def _load_hic(self):
        try:
            if '.hic' in self.hic_path:
                try:
                    import hicstraw
                    hic_obj = hicstraw.HiCFile(self.hic_path)
                except:
                    logging.warning(
                        'hicstraw is not installed, cannot load hic data, or the hic file is not found')
                    hic_obj = None
            elif 'cool' in self.hic_path:
                try:
                    import cooler
                    hic_obj = cooler.Cooler(self.hic_path+'::/resolutions/5000')
                except:
                    logging.warning(
                        'cooler is not installed, cannot load hic data, or the hic file is not found')
                    hic_obj = None
        except:
            logging.warning(
                'hic file is not found, or the file type is not supported')
            hic_obj = None
        return hic_obj

    def _subset_datasets(self):
        """
        Subset the zarr datasets based on the 
        - peak_name 
        - leave_out_celltypes and 
        - non_redundant and 
        - leave_out_chromosomes and
        - is_train
        parameters.
        """
        self.zarr_dict = {cdz.data_key: cdz for zarr_dir in self.zarr_dirs for cdz in [
            CelltypeDenseZarrIO(zarr_dir).subset_celltypes_with_data_name(self.peak_name)]}

        if self.keep_celltypes is not None and isinstance(self.keep_celltypes, list):
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update({data_key: cdz.leave_out_celltypes(
                    self.keep_celltypes, inverse=True)})
        elif isinstance(self.keep_celltypes, str):
            # remove the leave out celltypes using substring
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update({data_key: cdz.leave_out_celltypes_with_pattern(
                    self.keep_celltypes, inverse=True)})

        # remove the leave out celltypes using subset
        if self.leave_out_celltypes is not None and isinstance(self.leave_out_celltypes, list):
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update({data_key: cdz.leave_out_celltypes(
                    self.leave_out_celltypes, inverse=not self.is_train)})
        elif isinstance(self.leave_out_celltypes, str):
            # remove the leave out celltypes using substring
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update({data_key: cdz.leave_out_celltypes_with_pattern(
                    self.leave_out_celltypes, inverse=not self.is_train)})

        # remove redundant celltype instances, keep only one depth and one sample
        if self.non_redundant:
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update(
                    {data_key: cdz.non_redundant_celltypes(self.non_redundant)})

        # remove samples that do not meet minimum depth threshold
        if self.filter_by_min_depth:
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update(
                    {data_key: cdz.filter_by_min_depth(
                        self.filter_by_min_depth)}
                )

        # remove the leave out chromosomes
        if isinstance(self.leave_out_chromosomes, str):
            if ',' in self.leave_out_chromosomes:
                self.leave_out_chromosomes = self.leave_out_chromosomes.split(
                    ',')
            else:
                self.leave_out_chromosomes = [self.leave_out_chromosomes]

        if self.leave_out_chromosomes is not None and isinstance(self.leave_out_chromosomes, list):
            # subset the data
            for data_key, cdz in self.zarr_dict.items():
                self.zarr_dict.update({data_key: cdz.leave_out_chromosomes(
                    self.leave_out_chromosomes, inverse=not self.is_train)})
        return

    def calculate_metadata(self):
        """
        Calculate metadata for the dataset, including number of cell types, chunk size, number of chunks per chromosome, and total number of chunks.
        """
        first_zarr = next(iter(self.zarr_dict.values()))
        self.n_celltypes = sum(
            zarr.n_celltypes for zarr in self.zarr_dict.values())
        self.chunk_size = first_zarr.chunk_size
        self.chrom_n_chunks = first_zarr.chrom_n_chunks
        self.genome_chunk_length = sum(
            n_chunks - 1 for n_chunks in self.chrom_n_chunks.values())
        self.total_chunk_length = self.genome_chunk_length * self.n_celltypes

    def load_data(self, data_key, celltype_id, chr_name, start, end):
        """
        Load data from zarrdatapool
        """
        chr_chunk_idx = start // self.chunk_size
        celltype_peaks = self._query_peaks(
            celltype_id, chr_name, start, end, self.random_shift_peak)
        item_insulation = self._query_insulation(chr_name, start, end)

        track = self.zarr_dict[data_key].get_track(
            celltype_id, chr_name, start, end, sparse=True).T.astype(np.uint16)
        item_insulation = item_insulation.reset_index(drop=True).reset_index()

        celltype_peaks = celltype_peaks.reset_index(drop=True).reset_index()
        if self.motif_mean_std_obj is not None:
            try:
                motif_mean_std = self.motif_mean_std_obj.data_dict[chr_name][chr_chunk_idx:chr_chunk_idx+2].reshape(
                2, 2, -1).mean(0)
            except: # TODO need to check the best way to handle this!!!!!
                motif_mean_std = np.zeros((2, 2, 1))
        else:
            motif_mean_std = np.zeros((2, 2, 1))
        return chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std

    def load_window_data(self, window_index=None):
        """
        Load data for a single window.

        Parameters:
        window_index (int): The index of the window to load.

        Returns:
        tuple: A tuple containing the loaded data for the window, including window index, chromosome name, start and end positions, cell type ID, track data, peak sequence, and insulation data.

        Note:
        This method is used by PreloadDataPack to load data for a single window.
        """
        data_key, celltype_id = self._get_celltype_info(window_index)
        chr_name, chr_chunk_idx, start, end = self._get_chromosome_info(
            window_index)
        chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std = self.load_data(
            data_key, celltype_id, chr_name, start, end)
        item_insulation['key'] = str(
            window_index) + '_' + item_insulation['index'].astype(str)
        return window_index, chr_name, start, end, data_key, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std

    def _inactivated_peaks(self, celltype_peaks, peak_inactivation):
        """
        Generate a column label for 
        inactivated peaks that are in peak_inactivation use pyranges
        """
        # double reset_index to get numeric index
        celltype_peaks_ = pr(
            celltype_peaks.reset_index(drop=True).reset_index())
        peak_inactivation_ = pr(peak_inactivation)
        celltype_peaks_ = celltype_peaks_.join(
            peak_inactivation_, how='left', suffix='_peak').df
        # get numeric index of the peaks that are in peak_inactivation
        inactivated_peak_idx = celltype_peaks_.loc[celltype_peaks_[
            'Start_peak'] != -1]['index'].drop_duplicates().values
        return inactivated_peak_idx

    def _get_peak_names(self, data_key, celltype_id):
        """
        Return a list of peak names for a celltype, use glob peaks*
        """
        """Return a list of peak names for a celltype, use glob peaks*"""
        return [key for key in self.zarr_dict[data_key].dataset[celltype_id].keys() if 'peaks' in key]

    def _load_peaks(self, peak_name=None, peak_count_filter=0):
        """
        Load peaks data which is a dictionary of pandas dataframe feather
        """
        logging.info('Loading peaks data')
        peaks_dict = {}
        for data_key, cdz in self.zarr_dict.items():
            for celltype_id in cdz.ids:
                # check if the peak name exists in the zarr
                if peak_name not in self._get_peak_names(data_key, celltype_id):
                    continue

                peak = cdz.get_peaks(
                    celltype_id, peak_name)
                if self.max_peak_length is not None:
                    peak = peak.query(
                        'End-Start<@self.max_peak_length').reset_index(drop=True)

                # append negative peaks
                if self.negative_peak_name is not None and self.negative_peak_ratio > 0 and self.negative_peak_name in self._get_peak_names(data_key, celltype_id):
                    negative_peak = cdz.get_peaks(
                        celltype_id, self.negative_peak_name)
                    n_negative = int(self.negative_peak_ratio * peak.shape[0])
                    negative_peak = negative_peak.sample(n=n_negative)
                    peak = pd.concat([peak, negative_peak]
                                     ).reset_index(drop=True)
                    # sort the peaks
                    peak = peak.sort_values(
                        ['Chromosome', 'Start']).reset_index(drop=True)
                if self.center_expand_target != 0:
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
                # remove the peaks with too high or low Count
                lower_count_threshold = peak['Count'].quantile(0.0001)
                upper_count_threshold = peak['Count'].quantile(0.9999)
                peak = peak.query(
                    'Count>@lower_count_threshold and Count<@upper_count_threshold').reset_index(drop=True)
                peaks_dict[celltype_id] = pr(peak.query(
                    'Count>@peak_count_filter').reset_index(drop=True)).sort().df

        return peaks_dict

    def _load_insulation(self):
        """
        Load insulation data which is a list of pandas dataframe feather

        Parameters:
        insulation_paths (list): A list of paths to insulation data.

        Returns:
        pandas.DataFrame: A pandas dataframe containing insulation data.
        """
        logging.info('Loading insulation data')
        insulation_list = []
        for path in self.insulation_paths:
            insulation_list.append(pd.read_feather(path))

        insulation = pd.concat(insulation_list).drop_duplicates(
            subset=['Chromosome', 'Start', 'End'])
        if self.leave_out_chromosomes is not None and isinstance(self.leave_out_chromosomes, list):
            if self.is_train:
                # subset the data
                insulation = insulation.query(
                    'Chromosome not in @self.leave_out_chromosomes').reset_index(drop=True)
            else:
                insulation = insulation.query(
                    'Chromosome in @self.leave_out_chromosomes').reset_index(drop=True)
        return insulation.sample(frac=self.insulation_subsample_ratio).reset_index(drop=True)

    def _get_celltype_info(self, window_index):
        """
        Get the data_key and celltype from the window index

        Parameters:
        window_index (int): The index of the window.

        Returns:
        tuple: A tuple containing the data_key and celltype ID.
            """
        celltype_idx = window_index // self.genome_chunk_length
        data_key, celltype_id = self._get_celltype(celltype_idx)
        return data_key, celltype_id

    def _get_chromosome_info(self, window_index):
        """
        Get the chromosome name, chunk index, start and end positions from the window index

        Parameters:
        window_index (int): The index of the window.

        Returns:
        tuple: A tuple containing the chromosome name, chunk index, start and end positions.
        """
        chunk_idx = window_index % self.genome_chunk_length
        chr_name, chunk_idx = self._get_chr_chunk_idx(chunk_idx)
        start, end = self._get_start_end(chr_name, chunk_idx)
        return chr_name, chunk_idx, start, end

    def _generate_peak_sequence(self, celltype_peaks, sequence, window_start, window_end):
        """
        Generate peak sequence from peaks and sequence data. Peak sequence is a one-hot encoding of DNA.
        Places that are not peaks are masked with 0.

        Parameters:
        celltype_peaks (pandas.DataFrame): A pandas dataframe containing peak data.
        sequence (numpy.ndarray): A numpy array containing sequence data.
        window_start (int): The start position of the window.
        window_end (int): The end position of the window.

        Returns:
        scipy.sparse.csr_matrix: A sparse matrix containing peak sequence data.

        Note:
        This method is used by PreloadDataPack to generate peak sequence data.
        """
        peak_sequence = np.zeros_like(sequence, dtype=np.int8)
        for start, end in celltype_peaks:  # [['Start', 'End']].to_numpy()
            peak_sequence[start-window_start:end-window_start] = 1
        return csr_matrix(peak_sequence*sequence)

    def _query_peaks(self, celltype_id, chr_name, start, end, random_shift_peak=None):
        """
        Query peaks data for a celltype and a window.

        Parameters:
        celltype_id (str): The ID of the cell type.
        chr_name (str): The name of the chromosome.
        start (int): The start position of the window.
        end (int): The end position of the window.

        Returns:
        pandas.DataFrame: A pandas dataframe containing peak data.

        Note:
        This method is used by PreloadDataPack to query peaks data.
        """
        df = self.peaks_dict[celltype_id]
        if random_shift_peak is not None and isinstance(random_shift_peak, int) and random_shift_peak > 0:
            random_int = np.random.randint(-random_shift_peak,
                                           random_shift_peak, size=df.shape[0])
        else:
            random_int = 0
        df['Start'] = df['Start'].values + random_int
        df['End'] = df['End'].values + random_int
        df = df[(df.Chromosome == chr_name) & (df.Start >= start) & (df.End <= end)]
        return df

    def _query_insulation(self, chr_name, start, end):
        """
        Query insulation data for a window.

        Parameters:
        chr_name (str): The name of the chromosome.
        start (int): The start position of the window.
        end (int): The end position of the window.

        Returns:
        pandas.DataFrame: A pandas dataframe containing insulation data.

        Note:
        This method is used by PreloadDataPack to query insulation data.
        """
        return self.insulation.query(
            'Chromosome == @chr_name and Start >= @start and End <= @end')

    def _get_celltype(self, celltype_idx):
        """
        Get the data_key and celltype from the celltype index

        Parameters:
        celltype_idx (int): The index of the cell type.

        Returns:
        tuple: A tuple containing the data_key and celltype ID.
        """
        for data_key, cdz in self.zarr_dict.items():
            if celltype_idx < cdz.n_celltypes:
                return data_key, cdz.ids[celltype_idx]
            else:
                celltype_idx -= cdz.n_celltypes
        raise ValueError(f'Celltype index {celltype_idx} is out of range')

    def _get_data_key(self, celltype):
        """
        Get the data_key from the celltype ID

        Parameters:
        celltype (str): The ID of the cell type.

        Returns:
        str: The data_key.
        """
        return self.celltype_to_data_key[celltype]

    def _get_chr_chunk_idx(self, chunk_idx):
        """
        Get the chromosome name and chunk index from the chunk index

        Parameters:
        chunk_idx (int): The index of the chunk.

        Returns:
        tuple: A tuple containing the chromosome name and chunk index.
        """
        for chr_name, n_chunks in self.chrom_n_chunks.items():
            if chunk_idx < n_chunks-1:
                return chr_name, chunk_idx
            else:
                chunk_idx -= n_chunks-1
        raise ValueError(f'Chunk index {chunk_idx} is out of range')

    def _get_start_end(self, chr_name, chunk_idx):
        """
        Get the start and end position from the chunk index.
        Specifically, the start and end positions are the start and end positions of a (2*self.chunk_size) Mbp window.

        Parameters:
        chr_name (str): The name of the chromosome.
        chunk_idx (int): The index of the chunk.

        Returns:
        tuple: A tuple containing the start and end positions.
        """
        start = chunk_idx * self.chunk_size
        end = start + 2 * self.chunk_size
        return start, end

    def _get_window_index_from_chunk_idx(self, data_key, celltype_id, chunk_idx):
        """Get window index from data_key, celltype_id, chr and pos
        this is a reverse operation of _get_celltype_info and _get_chromosome_info
        [ celltype_1:(genome_chunks), celltype_2:(genome_chunks), ... ]
        first get the celltype index using celltype_id, 
        then get the chunk index using chr and pos, 
        then get the window index using celltype index and chunk index
        """
        celltype_idx = self._get_celltype_idx(data_key, celltype_id)
        return int(celltype_idx * self.genome_chunk_length + chunk_idx)

    def _get_celltype_idx(self, data_key, celltype_id):
        """Get celltype index from data_key and celltype_id"""
        return self.zarr_dict[data_key].ids.index(celltype_id)

    def _get_chunk_idx(self, chr, pos):
        """Get chunk index from chr and pos"""
        chr_chunk_idx = pos // self.chunk_size
        # determine how many chunks before the current chromosome, chr is string
        current_chr_idx = list(self.chrom_n_chunks.keys()).index(chr)
        chunk_idx = sum([self.chrom_n_chunks[list(self.chrom_n_chunks.keys())[
                        i]]-1 for i in range(current_chr_idx)]) + chr_chunk_idx
        return chunk_idx

    def generate_sample(self, chr_name, start, end, data_key, celltype_id,
                        mutations=None, peak_inactivation=None, padding=50):
        """
        Convenient handler for generate a single sample.
        """
        chr_name, start, end, celltype_id, track, _, celltype_peaks, motif_mean_std = self.load_data(
            data_key, celltype_id, chr_name, start, end)
        track_start = celltype_peaks['Start'].min() - padding
        track_end = celltype_peaks['End'].max() + padding
        assert track_start >= start and track_end <= end, f"Celltype peaks after padding is not within the input range {chr_name}:{start}-{end}"

        if peak_inactivation is not None:
            inactivated_peak_idx = self._inactivated_peaks(
                celltype_peaks, peak_inactivation)
        else:
            inactivated_peak_idx = None

        hic_matrix = 0
        if self.hic_obj is not None:
            hic_matrix = get_hic_from_idx(self.hic_obj, celltype_peaks)
            if hic_matrix is None:
                hic_matrix = np.zeros((len(celltype_peaks), len(celltype_peaks)))

        if self.additional_peak_columns is not None:
            # assume numeric columns
            additional_peak_features = celltype_peaks[self.additional_peak_columns].to_numpy(
            ).astype(np.float32)
        else:
            additional_peak_features = None

        assembly = self.assembly_dict[data_key]
        sequence = self.sequence[assembly].get_track(
            chr_name, track_start, track_end, sparse=False)
        if mutations is not None:
            # filter the mutation data with celltype_peaks
            mut_peak = pr(mutations.query('Chromosome==@chr_name')
                          ).join(pr(celltype_peaks)).df
            if mut_peak.shape[0] > 0:
                sequence_mut, sequence = get_sequence_with_mutations(
                    sequence, track_start, track_end, mut_peak)
                logging.info(
                    f"Mutated sequence for {chr_name}:{start}-{end} has been generated")
                logging.info(
                    f"Mutated sequence is different from the original sequence: {not np.array_equal(sequence, sequence_mut)}")
                sequence = sequence_mut

        celltype_peaks = celltype_peaks[[
            'Start', 'End']].to_numpy().astype(np.int64)

        sample_peak_sequence = self._generate_peak_sequence(
            celltype_peaks, sequence, track_start, track_end)

        # where the track locates in the window
        _start, _end = track_start - start, track_end - start
        # where the peaks locate in the track
        celltype_peaks = celltype_peaks - track_start

        sample_track = track[_start:_end]

        sample_peak_sequence = _stack_tracks_with_padding_and_inactivation(
            celltype_peaks, sample_peak_sequence, padding, inactivated_peak_idx)
        sample_track = _stack_tracks_with_padding_and_inactivation(
            celltype_peaks, sample_track, padding, inactivated_peak_idx)
        # remove atac and expression from inactivated peak
        if inactivated_peak_idx is not None:
            # keep the TSS column but set aTPM and expression to 0
            additional_peak_features[inactivated_peak_idx, 0:3] = 0

        sample = {
            'sample_track': sample_track, 'sample_peak_sequence': sample_peak_sequence,
            'celltype_peaks': celltype_peaks, 'motif_mean_std': motif_mean_std,
            'additional_peak_features': additional_peak_features, 'hic_matrix': hic_matrix,
            'metadata': {
                'celltype_id': celltype_id, 'chr_name': chr_name, 'libsize': self.zarr_dict[data_key].libsize[celltype_id],
                'start': start, 'end': end, 'i_start': _start, 'i_end': _end, 'mask_ratio': 0,
            }

        }

        return sample


class PreloadDataPack(object):
    """
    A class to store preloaded data for a slot.
    It will be initialized with a fixed number of windows. Each window contains a 4Mbp region of the genome.

    Parameters:
    preload_count (int): The number of windows to preload.
    zarr_data_pool (ZarrDataPool): A ZarrDataPool object.
    padding (int): The number of nucleotides to pad around each peak.
    mask_ratio (float): The ratio of nucleotides to mask in the peak sequence.
    n_peaks_lower_bound (int): The lower bound of number of peaks in a sample.
    n_peaks_upper_bound (int): The upper bound of number of peaks in a sample.
    window_index (numpy.ndarray): A numpy array containing the indices of the windows to preload.

    Returns:
    PreloadDataPack: A PreloadDataPack object.

    Note:
    This class is used by PretrainDataset to preload datax for a slot.
    """

    def __init__(self, preload_count: int, zarr_data_pool: ZarrDataPool, padding=50, mask_ratio=0.5,
                 n_peaks_lower_bound=5, n_peaks_upper_bound=200, n_peaks_sample_gap=50, use_insulation=True, window_index=None, peak_inactivation=None, mutations=None):
        # logging.info('Initializing PreloadDataPack')
        self.preload_count = preload_count
        self.zarr_data_pool = zarr_data_pool
        self.preloaded_data = []
        self.insulation_peak_counts = pd.DataFrame()
        self.preloaded_data_window_indices_mapping = {}
        self.next_sample = 0
        self.padding = padding
        self.peak_inactivation = peak_inactivation
        self.mutations = mutations
        self.mask_ratio = mask_ratio
        self.n_peaks_lower_bound = n_peaks_lower_bound
        self.n_peaks_upper_bound = n_peaks_upper_bound
        self.n_peaks_sample_gap = n_peaks_sample_gap
        self.use_insulation = use_insulation
        self.additional_peak_columns = self.zarr_data_pool.additional_peak_columns
        if window_index is None:
            window_index = np.random.randint(
                0, self.zarr_data_pool.total_chunk_length, size=self.preload_count)
        self.window_index = window_index
        self.preload_data(window_index)

        # logging.info('PreloadDataPack initialized')

    def __len__(self):
        """
        Return the number of samples in the preload data pack.
        """
        return self._calculate_total_samples()

    # def __iter__(self):
    #     return self

    def get_next_sample(self):
        """
        Get the next sample from the preload data pack.

        Returns:
        tuple: A tuple containing the extracted sample track, peak sequence, and a dictionary with metadata
        """
        if self.next_sample >= self.__len__():
            return None
        else:
            sample = self.get_sample_with_idx(self.next_sample)
            self.next_sample += 1
            return sample

    def preload_data(self, window_index=None):
        """
        Preload data for a fixed number of windows. window_index is a numpy array containing the indices of the windows to preload.

        Parameters:
        window_index (numpy.ndarray): A numpy array containing the indices of the windows to preload.
        """
        # Preload data
        # Load data for a fixed number of windows
        self.preloaded_data_window_indices_mapping = {}
        for i, window_index in enumerate(window_index):
            self.preloaded_data.append(
                self.zarr_data_pool.load_window_data(window_index))
            self.preloaded_data_window_indices_mapping[window_index] = i
        # trigger the computation of peak_num_per_sample
        self._calculate_peak_num_per_sample()
        self._calculate_window_peak_counts()
        if self.insulation_peak_counts.shape[0] == 0 and self.use_insulation:
            logging.info('No valid insulation peak count')
            return PreloadDataPack(self.preload_count, self.zarr_data_pool, self.padding, self.mask_ratio, self.n_peaks_lower_bound, self.n_peaks_upper_bound, self.n_peaks_sample_gap, self.use_insulation, window_index, self.peak_inactivation, self.mutations)

    def get_sample_with_idx(self, idx):
        """
        Get a sample from the preload data pack using the index.

        Parameters:
        idx (int): The index of the sample.
        use_insulation (bool): Whether to use insulation data to select the sample.

        Returns:
        tuple: A tuple containing the extracted sample track, peak sequence, and a dictionary with metadata
        """
        if self.use_insulation:
            return self._get_sample_with_key(self.insulation_peak_counts.iloc[idx]['key'])
        else:
            return self._get_sample_from_peak(idx)

    def _get_sample_from_peak(self, idx):
        """
        Get a sample from the preloaded data pack using the index. 
        This is used when use_insulation is False. We will non-overlappingly slide across the window to get a sample with n_peaks_upper_bound peaks.
        """
        # find the window index based on self.per_window_n_samples
        window_slot = np.argmax(self.per_window_n_samples_cumsum > idx)
        # get the window
        peak_index = idx - \
            self.per_window_n_samples_cumsum[window_slot -
                                             1] if window_slot > 0 else idx
        # get the samples
        return self._extract_sample_from_window_without_insulation(self.preloaded_data[window_slot], peak_index)

    def _get_sample_with_key(self, key):
        """
        Get a sample from the preload data pack using the key.

        Parameters:
        key (str): The key of the sample.

        Returns:
        tuple: A tuple containing the extracted sample track, peak sequence, and a dictionary with metadata"""
        window_index = int(key.split('_')[0])
        insulation_index = int(key.split('_')[1])
        window_slot = self.preloaded_data_window_indices_mapping[window_index]
        return self._extract_sample_from_window(self.preloaded_data[window_slot], insulation_index)

    def _extract_sample_from_window_without_insulation(self, window, peak_index, peak_start=None, peak_end=None, mut=None):
        """
        Extract a single sample from a preloaded window without using insulation data.

        Parameters:
        window (tuple): Contains preloaded data for a window, including window index, chromosome name,
                        start and end positions, cell type ID, track data, peak sequence, and insulation data.
        peak_index (int): The index of the peak to sample from.

        Returns:
        tuple: A tuple containing the extracted sample track, peak sequence, and a dictionary with metadata
            about the extracted sample, including cell type ID, chromosome name, and positions.
        """
        window_index, chr_name, start, end, data_key, celltype_id, track, insulations, celltype_peaks, motif_mean_std = window

        if peak_start is None or peak_end is None:
            peak_start = peak_index * self.n_peaks_sample_gap
            peak_end = peak_start + self.n_peaks_upper_bound
        celltype_peaks = celltype_peaks.iloc[peak_start:peak_end]
        track_start = celltype_peaks['Start'].min() - self.padding
        track_end = celltype_peaks['End'].max() + self.padding
        return self._generate_sample(chr_name, start, end, data_key, celltype_id, track, celltype_peaks, motif_mean_std,
                                     track_start, track_end, mut)

    def _inactivated_peaks(self, celltype_peaks, peak_inactivation):
        """
        Generate a column label for 
        inactivated peaks that are in peak_inactivation use pyranges
        """
        # double reset_index to get numeric index
        celltype_peaks_ = pr(
            celltype_peaks.reset_index(drop=True).reset_index())
        peak_inactivation_ = pr(peak_inactivation)
        celltype_peaks_ = celltype_peaks_.join(
            peak_inactivation_, how='left', suffix='_peak')
        # get numeric index of the peaks that are in peak_inactivation
        inactivated_peak_idx = celltype_peaks_.loc[celltype_peaks_[
            'Start_peak'] != -1]['index'].drop_duplicates()
        return inactivated_peak_idx

    def _generate_sample(self, chr_name, start, end, data_key, celltype_id, track, celltype_peaks, motif_mean_std,
                         track_start, track_end, mutations=None):
        """
        Generate a single sample from a window.
        """
        # peak_inactivation is a dataframe of peaks to inactivate
        # overlap with celltype_peaks to keep the peaks that are not in peak_inactivation, unless the peak is a TSS
        if self.peak_inactivation is not None and self.peak_inactivation != 'random_tss':
            inactivated_peak_idx = self._inactivated_peaks(
                celltype_peaks, self.peak_inactivation)
        elif self.peak_inactivation == 'random_tss':
            inactivated_peak_idx = celltype_peaks.reset_index(
                drop=True).reset_index().query('TSS==1').sample(frac=0.1).index.values
        else:
            inactivated_peak_idx = None
        if self.additional_peak_columns is not None:
            # assume numeric columns
            additional_peak_features = celltype_peaks[self.additional_peak_columns].to_numpy(
            ).astype(np.float32)
        else:
            additional_peak_features = None

        assembly = self.zarr_data_pool.assembly_dict[data_key]
        sequence = self.zarr_data_pool.sequence[assembly].get_track(
            chr_name, track_start, track_end, sparse=False)

        if mutations is not None:
            # filter the mutation data with celltype_peaks
            mut_peak = pr(mutations.query('Chromosome==@chr_name')
                          ).join(pr(celltype_peaks)).df
            if mut_peak.shape[0] > 0:
                sequence_mut, sequence = get_sequence_with_mutations(
                    sequence, track_start, track_end, mut_peak)
                sequence = sequence_mut

        hic_matrix = None
        if self.zarr_data_pool.hic_obj is not None:
            hic_matrix = get_hic_from_idx(
                self.zarr_data_pool.hic_obj, celltype_peaks)
            if hic_matrix is None:
                hic_matrix = np.zeros(
                    (self.n_peaks_upper_bound, self.n_peaks_upper_bound))
                
        celltype_peaks = celltype_peaks[[
            'Start', 'End']].to_numpy().astype(np.int64)

        sample_peak_sequence = self.zarr_data_pool._generate_peak_sequence(
            celltype_peaks, sequence, track_start, track_end)

        # where the track locates in the window
        _start, _end = track_start - start, track_end - start
        # where the peaks locate in the track
        celltype_peaks = celltype_peaks - track_start

        sample_track = track[_start:_end]

        sample_peak_sequence = _stack_tracks_with_padding_and_inactivation(
            celltype_peaks, sample_peak_sequence, self.padding, inactivated_peak_idx)
        sample_track = _stack_tracks_with_padding_and_inactivation(
            celltype_peaks, sample_track, self.padding, inactivated_peak_idx)
        # remove atac and expression from inactivated peak
        if inactivated_peak_idx is not None:
            # keep the TSS column but set aTPM and expression to 0
            additional_peak_features[inactivated_peak_idx, 0:3] = 0

        sample = {
            'sample_track': sample_track, 'sample_peak_sequence': sample_peak_sequence,
            'celltype_peaks': celltype_peaks, 'motif_mean_std': motif_mean_std,
            'additional_peak_features': additional_peak_features, 'hic_matrix': hic_matrix,
            'metadata': {
                'celltype_id': celltype_id, 'chr_name': chr_name, 'libsize': self.zarr_data_pool.zarr_dict[data_key].libsize[celltype_id],
                'start': start, 'end': end, 'i_start': _start, 'i_end': _end, 'mask_ratio': self.mask_ratio
            }

        }

        return sample

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
        window_index, chr_name, start, end, data_key, celltype_id, track, insulations, celltype_peaks, motif_mean_std = window
        if len(insulations) == 0:
            raise ValueError('Empty insulation')
        track_start, track_end = self._insulation_sampler(
            insulations, insulation_index)
        celltype_peaks = celltype_peaks.query(
            'Start>@track_start and End<@track_end')
        if celltype_peaks.shape[0] < self.n_peaks_lower_bound:
            logging.info('No enough peaks')
        return self._generate_sample(chr_name, start, end, data_key, celltype_id, track, celltype_peaks, motif_mean_std,
                                     track_start, track_end)

    def _insulation_sampler(self, insulation_df, insulation_index=None):
        """
        Sample insulation from the insulation dataframe

        Parameters:
        insulation_df (pandas.DataFrame): A pandas dataframe containing insulation data.
        insulation_index (int): The index of the insulation data to sample.

        Returns:
        tuple: A tuple containing the start and end positions of the sampled insulation data.
        """
        if insulation_index is None:
            insulation_index = np.random.randint(0, len(insulation_df))
        insulation_start, insulation_end = insulation_df.iloc[insulation_index][[
            'Start', 'End']]
        return insulation_start, insulation_end

    def _calculate_peak_num_per_sample(self):
        """
        Calculate the number of peaks per sample.

        Returns:
        pandas.DataFrame: A pandas dataframe containing the number of peaks per sample.
        """
        insulation_pool_peak_num = {}
        for window_index, chr_name, start, end, data_key, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std in self.preloaded_data:
            try:
                insulation_pool_peak_num.update(
                    self._get_peak_count(item_insulation, celltype_peaks))
            except:
                continue
        # Convert to DataFrame and sort
        insulation_pool_peak_num_df = pd.DataFrame.from_dict(
            insulation_pool_peak_num, orient='index').reset_index()
        if insulation_pool_peak_num_df.shape[0] == 0:
            return pd.DataFrame(columns=['key', 'peak_num'])
        insulation_pool_peak_num_df.columns = ['key', 'peak_num']
        # insulation_pool_peak_num_df.sort_values('peak_num', inplace=True)
        # shuffle the dataframe
        # avoid boundary case by adding 2 to lower bound
        insulation_pool_peak_num_df = insulation_pool_peak_num_df.query(
            'peak_num>=@self.n_peaks_lower_bound and peak_num<=@self.n_peaks_upper_bound')
        insulation_pool_peak_num_df = insulation_pool_peak_num_df.sample(
            frac=1).reset_index(drop=True)
        self.insulation_peak_counts = insulation_pool_peak_num_df
        return insulation_pool_peak_num_df

    def _calculate_window_peak_counts(self):
        """
        Calculate the number of peaks per preloaded window.
        """
        window_peak_counts = {}
        per_window_n_samples = []

        for window_index, chr_name, start, end, data_key, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std in self.preloaded_data:
            per_window_n_samples.append(
                (celltype_peaks.shape[0] - self.n_peaks_upper_bound) // self.n_peaks_sample_gap)
            window_peak_counts[window_index] = celltype_peaks.shape[0]
        self.window_peak_counts = window_peak_counts
        self.per_window_n_samples = np.array(per_window_n_samples)
        self.per_window_n_samples_cumsum = np.cumsum(self.per_window_n_samples)

    def _calculate_total_samples(self):
        """
        Calculate the total number of samples in the preload data pack when not using insulation data.
        """
        if self.use_insulation:
            return self.insulation_peak_counts.shape[0]
        else:
            return sum(self.per_window_n_samples)

    def _get_peak_count(self, item_insulation, celltype_peaks):
        """
        Get the number of peaks per sample.

        Parameters:
        item_insulation (pandas.DataFrame): A pandas dataframe containing insulation data.
        celltype_peaks (pandas.DataFrame): A pandas dataframe containing peak data.

        Returns:
        dict: A dictionary containing the number of peaks per sample.
        """
        df = pr(item_insulation).join(pr(celltype_peaks), suffix="_peak", apply_strand_suffix=True).df.groupby(
            'key').index_peak.count().reset_index()
        return df.set_index('key').to_dict()['index_peak']


class PretrainDataset(Dataset):
    def __init__(self,
                 is_train=True,
                 sequence_obj=None,
                 zarr_dirs=None,
                 genome_seq_zarr=None,
                 genome_motif_zarr=None,
                 use_insulation=True,
                 insulation_paths=[],
                 insulation_subsample_ratio=0.1,
                 hic_path=None,

                 peak_name='peaks',
                 additional_peak_columns=None,
                 max_peak_length=None,
                 center_expand_target=None,
                 peak_count_filter=0,
                 n_peaks_lower_bound=5,
                 n_peaks_upper_bound=200,
                 n_peaks_sample_gap=50,

                 non_redundant=False,
                 filter_by_min_depth=False,

                 preload_count=50,
                 n_packs=1,

                 padding=50,
                 mask_ratio=0.5,
                 negative_peak_name=None,
                 negative_peak_ratio=0,
                 random_shift_peak=True,
                 peak_inactivation=None,
                 mutations=None,

                 keep_celltypes=None,
                 leave_out_celltypes=None,
                 leave_out_chromosomes=None,
                 dataset_size=655_36,
                 **kwargs
                 ):
        super().__init__()
        """
        Pretrain dataset for GET model.

        This dataset is used to train the GET model in a self-supervised manner.
        It loads data from a set of zarr files and generates samples for training.
        Each sample consists of a track and a peak sequence extracted from a 4Mbp window.
        The track is a 2D sparse array of shape (nucleotide,1) and the peak sequence is a 2D array of shape (nucleotide,4). 
        Peak sequence is generated by multiplying the peak mask with the genomic sequence. 
        Track is cell-type specific ATAC-seq insertion counts per nucleotide, without any normalization. 
        Peak sequence is a one-hot encoding of DNA. Other metadata about the sample is also included in the sample as a dictionary.

        Parameters:
        is_train (bool): Whether the dataset is used for training.
        sequence_obj (DenseZarrIO): A DenseZarrIO object containing sequence data.
        zarr_dirs (list): A list of zarr directories containing peak data.
        genome_seq_zarr (str): The path to the zarr file containing genome sequence data.
        genome_motif_zarr (str): The path to the zarr file containing motif mean and standard deviation data.
        use_insulation (bool): Whether to use insulation data to select samples.
        insulation_paths (list): A list of paths to insulation data.
        insulation_subsample_ratio (float): The ratio of insulation data to subsample.
        hic_path (str): The path to the Hi-C data.

        peak_name (str): The name of the peak data.
        additional_peak_columns (list): A list of additional peak columns to include in the sample.
        max_peak_length (int): The maximum length of a peak.
        center_expand_target (int): The target length to expand the center of the peak.
        n_peaks_lower_bound (int): The lower bound of number of peaks in a sample.
        n_peaks_upper_bound (int): The upper bound of number of peaks in a sample.
        n_peaks_sample_gap (int): The gap between samples.
        non_redundant (bool): Whether to remove redundant samples.
        filter_by_min_depth (bool): Whether to filter samples by minimum depth.

        preload_count (int): The number of windows to preload.
        n_packs (int): The number of data packs to preload.

        padding (int): The number of nucleotides to pad around each peak.
        mask_ratio (float): The ratio of nucleotides to mask in the peak sequence.
        negative_peak_name (str): The name of the negative peak data.
        negative_peak_ratio (float): The ratio of negative peaks to sample.
        random_shift_peak (bool): Whether to randomly shift peaks.
        peak_inactivation (pandas.DataFrame): A pandas dataframe containing peaks to inactivate.
        mutations (pandas.DataFrame): A pandas dataframe containing mutations to apply.

        keep_celltypes (list): A list of cell types to keep.
        leave_out_celltypes (list): A list of cell types to leave out.
        leave_out_chromosomes (list): A list of chromosomes to leave out.
        dataset_size (int): The size of the dataset.

        Returns:
        PretrainDataset: A PretrainDataset object.
        """
        logging.info('Initializing PretrainDataset')
        # log all parameters
        for key, value in locals().items():
            logging.info(f'{key}: {value}')
        self.preload_count = preload_count
        self.padding = padding
        self.mask_ratio = mask_ratio
        self.peak_name = peak_name
        self.negative_peak_name = negative_peak_name
        self.insulation_subsample_ratio = insulation_subsample_ratio
        self.n_peaks_lower_bound = n_peaks_lower_bound
        self.n_peaks_upper_bound = n_peaks_upper_bound
        self.n_peaks_sample_gap = n_peaks_sample_gap
        self.peak_count_filter = peak_count_filter
        self.negative_peak_ratio = negative_peak_ratio
        self.random_shift_peak = random_shift_peak
        self.use_insulation = use_insulation
        self.keep_celltypes = keep_celltypes
        self.leave_out_celltypes = leave_out_celltypes
        self.leave_out_chromosomes = leave_out_chromosomes
        self.is_train = is_train
        self.non_redundant = non_redundant
        self.filter_by_min_depth = filter_by_min_depth
        self.dataset_size = dataset_size
        self.n_packs = n_packs
        self.additional_peak_columns = additional_peak_columns
        self.peak_inactivation = peak_inactivation
        self.mutations = mutations
        self.hic_path = hic_path
        # ensure use_insulation is False if hic_path is not None
        if self.hic_path is not None:
            logging.info(
                'hic_path is not None, use_insulation is set to False')
            self.use_insulation = False
        self.sequence = sequence_obj if sequence_obj is not None else get_sequence_obj(
            genome_seq_zarr)
        if genome_motif_zarr is None:
            self.mms=None
        else:
            self.mms = MotifMeanStd(genome_motif_zarr)
        self.datapool = ZarrDataPool(zarr_dirs=zarr_dirs, genome_seq_zarr=genome_seq_zarr,
                                     insulation_paths=insulation_paths, peak_name=peak_name,
                                     negative_peak_name=negative_peak_name,
                                     insulation_subsample_ratio=self.insulation_subsample_ratio, max_peak_length=max_peak_length, center_expand_target=center_expand_target, sequence_obj=self.sequence,
                                     motif_mean_std_obj=self.mms,
                                     additional_peak_columns=self.additional_peak_columns,
                                     keep_celltypes=self.keep_celltypes,
                                     leave_out_celltypes=self.leave_out_celltypes,
                                     leave_out_chromosomes=self.leave_out_chromosomes,
                                     peak_count_filter=self.peak_count_filter,
                                     is_train=self.is_train, non_redundant=self.non_redundant, filter_by_min_depth=self.filter_by_min_depth,
                                     negative_peak_ratio=self.negative_peak_ratio, random_shift_peak=self.random_shift_peak, hic_path=self.hic_path)

        # initialize n_packs preload data packs
        self.preload_data_packs = None
        self.current_pack = 0
        self.avaliable_packs = list(range(n_packs))

    def __getitem__(self, index: int):
        if self.preload_data_packs is None:
            self.preload_data_packs = [PreloadDataPack(
                preload_count=self.preload_count, zarr_data_pool=self.datapool, padding=self.padding, mask_ratio=self.mask_ratio, n_peaks_lower_bound=self.n_peaks_lower_bound, n_peaks_upper_bound=self.n_peaks_upper_bound, n_peaks_sample_gap=self.n_peaks_sample_gap, use_insulation=self.use_insulation, peak_inactivation=self.peak_inactivation, mutations=self.mutations) for _ in range(self.n_packs)]
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

    def reload_data(self, slot, window_index=None):
        # This method runs in a separate thread
        # logging.info(f'Async reloading data for slot {index}')
        # reload by reinitializing the preload data pack and put it back to the preload_data_packs
        self.preload_data_packs[slot] = PreloadDataPack(
            preload_count=self.preload_count, zarr_data_pool=self.datapool, padding=self.padding, mask_ratio=self.mask_ratio, n_peaks_lower_bound=self.n_peaks_lower_bound, n_peaks_upper_bound=self.n_peaks_upper_bound, n_peaks_sample_gap=self.n_peaks_sample_gap, use_insulation=self.use_insulation, window_index=window_index, peak_inactivation=self.peak_inactivation, mutations=self.mutations)
        # self.preload_data_packs[slot].preload_data()
        # add the index back to avaliable packs
        self.avaliable_packs.append(slot)

    def debug_getitem(self, index):
        self.preload_data_packs = [0]
        self.reload_data(0)
        return self._getitem(index)


def worker_init_fn_get(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(torch.initial_seed() + worker_id)

    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.preload_data_packs is None:
        dataset.preload_data_packs = [PreloadDataPack(
            preload_count=dataset.preload_count, zarr_data_pool=dataset.datapool, padding=dataset.padding, mask_ratio=dataset.mask_ratio, n_peaks_lower_bound=dataset.n_peaks_lower_bound, n_peaks_upper_bound=dataset.n_peaks_upper_bound, n_peaks_sample_gap=dataset.n_peaks_sample_gap, use_insulation=dataset.use_insulation, peak_inactivation=dataset.peak_inactivation, mutations=dataset.mut) for _ in range(dataset.n_packs)]


class InferenceDataset(PretrainDataset):
    MAX_CONTEXT_DISTANCE = 4000000
    PROMOTER_EXTEND = 300

    def __init__(self, assembly, gencode_obj, gene_list=None, **kwargs):
        super().__init__(**kwargs)
        self.gencode_obj = gencode_obj[assembly]
        if isinstance(gene_list, str):
            if ',' in gene_list:
                gene_list = gene_list.split(',')
            elif os.path.exists(gene_list):
                gene_list = np.loadtxt(gene_list, dtype=str)
        self.gene_list = gene_list if gene_list is not None else self.gencode_obj.gtf['gene_name'].unique(
        )
        self.tss_chunk_idx = self._generate_tss_chunk_idx()
        self.accessible_genes
        self.gene_celltype_pair

    @property
    def accessible_genes(self):
        """Find overlap between self.tss_chunk_idx and self.datapool.peaks_dict"""
        if not hasattr(self, '_accessible_genes'):
            _accessible_genes = {}
            for key, peak in self.datapool.peaks_dict.items():
                _join = pr(self.tss_chunk_idx).join(
                    pr(peak)).df
                if _join.empty:
                    continue
                else:
                    _accessible_genes[key] = _join['gene_name'].unique()
            if self.gene_list is not None:
                _accessible_genes = {key: np.intersect1d(
                    genes, self.gene_list) for key, genes in _accessible_genes.items()}
            self._accessible_genes = _accessible_genes
        return self._accessible_genes

    @property
    def gene_celltype_pair(self):
        """get pairs of accessible genes and celltype as a list"""
        if not hasattr(self, '_gene_celltype_pair'):
            gene_celltype_pair = []
            for celltype_id, genes in self.accessible_genes.items():
                for gene in genes:
                    # validate gene in celltype
                    data_key = self.datapool.celltype_to_data_key[celltype_id]
                    info = self._get_window_idx_for_gene_and_celltype(
                        data_key, celltype_id, gene)
                    window_idx = info['window_idx'][0]
                    gene_info = self._get_gene_info_from_window_idx(
                        window_idx).query('gene_name==@gene')
                    if gene_info.shape[0] == 0:
                        continue
                    gene_celltype_pair.append((celltype_id, gene))

            self._gene_celltype_pair = gene_celltype_pair
        return self._gene_celltype_pair

    def _generate_tss_chunk_idx(self):
        """Determine the windows to extract for each gene"""
        # if os.path.exists(self.gencode_obj.feather_file.replace('.feather', '_tss_chunk_idx.feather')):
        #     self.tss_chunk_idx = pd.read_feather(
        #         self.gencode_obj.feather_file.replace('.feather', '_tss_chunk_idx.feather'))
        #     return self.tss_chunk_idx

        self.tss_chunk_idx = self.gencode_obj.gtf.query(
            'gene_name in @self.gene_list').copy()
        for i, row in self.tss_chunk_idx.iterrows():
            # get window_index for each gene
            gene_chr = row.Chromosome
            gene_start = row.Start
            self.tss_chunk_idx.loc[i, 'chunk_idx'] = self.datapool._get_chunk_idx(
                gene_chr, gene_start)

        # save the tss_chunk_idx as feather in the same directory as the gencode file
        self.tss_chunk_idx.to_feather(self.gencode_obj.feather_file.replace(
            '.feather', '_tss_chunk_idx.feather'))
        return self.tss_chunk_idx

    def _get_window_idx_for_tss_and_celltype(self, data_key, celltype_id, tss_idx):
        """Get window index for a gene and celltype"""
        chunk_idx = self.tss_chunk_idx.loc[tss_idx, 'chunk_idx']
        gene_name = self.tss_chunk_idx.loc[tss_idx, 'gene_name']
        tss_coord = self.tss_chunk_idx.loc[tss_idx, 'Start']
        strand = self.tss_chunk_idx.loc[tss_idx, 'Strand']
        strand = 0 if strand == '+' else 1
        return {
            'data_key': data_key,
            'celltype_id': celltype_id,
            'gene_name': gene_name,
            'window_idx': np.array([self.datapool._get_window_index_from_chunk_idx(data_key, celltype_id, chunk_idx)]),
            'chr_name': self.tss_chunk_idx.loc[tss_idx, 'Chromosome'],
            'tss_coord': tss_coord,
            'strand': strand}

    def _get_window_idx_for_gene_and_celltype(self, data_key, celltype_id, gene_name):
        """Get window index for a gene and celltype"""
        gene_df = self.tss_chunk_idx.query('gene_name==@gene_name')
        chunk_idxs = gene_df['chunk_idx']
        strand = gene_df['Strand'].values[0]
        tss_coord = gene_df['Start'].values
        if strand == '-' or strand == 1:
            tss_coord = tss_coord[-1]
            strand = 1
        elif strand == '+' or strand == 0:
            tss_coord = tss_coord[0]
            strand = 0

        return {
            'data_key': data_key,
            'celltype_id': celltype_id,
            'gene_name': gene_name,
            'window_idx': np.unique([self.datapool._get_window_index_from_chunk_idx(data_key, celltype_id, chunk_idx) for chunk_idx in chunk_idxs]),
            'chr_name': gene_df['Chromosome'].values[0],
            'tss_coord': tss_coord,
            'strand': strand}

    def _get_gene_info_from_window_idx(self, window_idx):
        # TODO: genome_chunk_length is affected by leave out chromosome by tss_chunk_idx is not
        chunk_idx = window_idx % (self.tss_chunk_idx.chunk_idx.max() + 1)
        gene_df = self.tss_chunk_idx.query(
            'chunk_idx==@chunk_idx or chunk_idx==@chunk_idx+1')
        return gene_df

    def __len__(self):
        return len(self.gene_celltype_pair)

    def __getitem__(self, idx):
        celltype_id, gene_name = self.gene_celltype_pair[idx]
        data_key = self.datapool.celltype_to_data_key[celltype_id]
        return self.get_item_for_gene_in_celltype(data_key, celltype_id, gene_name, self.mutations, self.peak_inactivation)

    def get_item_for_gene_in_celltype(self, data_key, celltype_id, gene_name, track_start=None, track_end=None, mutations=None, peak_inactivation=None):
        info = self._get_window_idx_for_gene_and_celltype(
            data_key, celltype_id, gene_name)
        window_idx = info['window_idx'][0]
        gene_info = self._get_gene_info_from_window_idx(
            window_idx).query('gene_name==@gene_name')
        if gene_info.shape[0] == 0:
            return None
        return self._get_item_for_gene_in_celltype(mutations, peak_inactivation, track_start, track_end, gene_info, info)

    def _get_item_for_gene_in_celltype(self, mutations, peak_inactivation, track_start, track_end, gene_info, info):
        celltype_id = info['celltype_id']
        chr_name = info['chr_name']
        data_key = info['data_key']

        # get the neighboring peaks for the gene
        info, peaks_in_locus, track_start, track_end = self._calculate_peak_bounds_for_gene(
            gene_info, info, track_start, track_end)
        
        # generate sample
        sample = self.datapool.generate_sample(
            chr_name, track_start, track_end, data_key, celltype_id, mutations=mutations, peak_inactivation=peak_inactivation, padding=self.padding)

        # get the peak index for mutations in the sample
        self._get_peak_idx_for_mutations(mutations, peaks_in_locus, info)

        # update sample metadata with new information
        sample['metadata'].update(info)

        return sample

    def _get_peak_idx_for_mutations(self, mutations, peaks_in_locus, info):
        """Get the peak index for mutations in the sample."""
        mut_peak = None
        if mutations is not None:
            mut_peak = pr(peaks_in_locus).join(pr(mutations)).df
            if mut_peak.shape[0] > 0:
                mut_peak = mut_peak['index'].values - info['peak_start']
        info['mut_peak'] = mut_peak

    def _calculate_peak_bounds_for_gene(self, gene_info, info, track_start=None, track_end=None):
        """
        Calculate the peak bounds for a specific gene.
        """
        celltype_id = info['celltype_id']
        chr_name = info['chr_name']
        tss_coord = info['tss_coord']
        gene_name = info['gene_name']
        # get peaks in gene locus
        peaks_in_locus = self.get_peaks_around_pos(
            celltype_id, chr_name, tss_coord)

        # get the absolute peak positions
        gene_df, tss_peak, all_tss_peak = self._get_absolute_tss_peak(
            gene_info, peaks_in_locus, gene_name)
        relative_all_tss_peak = all_tss_peak - tss_peak
        # get the relative peak positions and track bounds if not provided
        peaks_in_locus, track_start, track_end, tss_peak, peak_start, original_peak_start = self._get_relative_coord_and_idx(
            peaks_in_locus, track_start, track_end, gene_name, gene_df, tss_peak)
        all_tss_peak = np.unique(relative_all_tss_peak + tss_peak)
        info.update({'track_start': track_start, 'track_end': track_end,
                     'tss_peak': tss_peak, 'all_tss_peak': all_tss_peak, 'peak_start': peak_start, 'original_peak_start': original_peak_start})
        return info, peaks_in_locus, track_start, track_end

    def _get_relative_coord_and_idx(self, peaks_in_locus, track_start, track_end, gene_name, gene_df, tss_peak):
        """
        Get the relative peak positions and track bounds for a specific gene.
        """
        if track_start is None or track_end is None:
            # Get the peak start and end positions based on n_peaks_upper_bound
            peak_start = max(0, tss_peak - self.n_peaks_upper_bound // 2)
            peak_end = peak_start + self.n_peaks_upper_bound
            if peak_end > peaks_in_locus.shape[0]:
                peak_end = peaks_in_locus.shape[0]
                peak_start = max(0, peak_end - self.n_peaks_upper_bound)
            tss_peak = tss_peak - peak_start
            track_start = peaks_in_locus.iloc[peak_start].Start-self.padding
            track_end = peaks_in_locus.iloc[peak_end-1].End+self.padding
            original_peak_start = peaks_in_locus.iloc[peak_start].original_peak_index
        else:
            peaks_in_locus_subset = peaks_in_locus.query(
                'Chromosome==@chr_name').query('Start>=@track_start & End<=@track_end')
            if peaks_in_locus_subset.shape[0] == 0:
                raise ValueError(
                    f"No peaks found in the specified region for gene {gene_name}")
            peak_start, peak_end = peaks_in_locus_subset.index.min(
            ), peaks_in_locus_subset.index.max()
            tss_peak = pr(peaks_in_locus_subset).join(
                pr(gene_df)).df['index'].values
            tss_peak = tss_peak - peak_start
            peaks_in_locus = peaks_in_locus_subset
            original_peak_start = peaks_in_locus.iloc[peak_start].original_peak_index
        
        if isinstance(original_peak_start, pd.Series):
            original_peak_start = original_peak_start.values[0]
        if isinstance(peak_start, pd.Series):
            peak_start = peak_start.values[0]
        if isinstance(tss_peak, pd.Series):
            tss_peak = tss_peak.values[0]
        if isinstance(track_start, pd.Series):
            track_start = track_start.values[0]
        if isinstance(track_end, pd.Series):
            track_end = track_end.values[0]
        return peaks_in_locus, track_start, track_end, tss_peak, peak_start, original_peak_start

    def _get_absolute_tss_peak(self, gene_info, peaks_in_locus, gene_name):
        """
        Get the absolute tss peak index for a specific gene (tss_peak) and the peaks in the gene locus (gene_df).
        """
        columns_to_include = ['index', 'Chromosome', 'Start',
                              'End',  'gene_name', 'Strand', 'chunk_idx']
        if self.additional_peak_columns is not None:
            columns_to_include += self.additional_peak_columns
        gene_info_copy = gene_info.copy().query('gene_name==@gene_name')
        if gene_info_copy.shape[0] == 0:
            print(
                f"Gene {gene_name} not found in the gene information, removing and skipping it.")
        gene_df = pr(peaks_in_locus.copy().reset_index()).join(pr(gene_info_copy[['Chromosome', 'Start', 'End', 'gene_name', 'Strand', 'chunk_idx']].drop_duplicates(
        )).extend(self.PROMOTER_EXTEND), suffix="_gene", how='left', apply_strand_suffix=False).df[columns_to_include].set_index('index')
        gene_df = gene_df.query('gene_name==@gene_name')
        if gene_df.shape[0] == 0:
            print(
                f"Gene {gene_name} not found in the peak information, removing and skipping it.")
            return
        elif gene_df.shape[0] > 1:
            strand = gene_df.Strand.values[0]
            all_tss_peak = gene_df.index.values
            tss_peak = all_tss_peak[0] if strand == '+' else all_tss_peak[-1]
        else:
            strand = gene_df.Strand.values
            tss_peak = gene_df.index.values
            all_tss_peak = [tss_peak]
        return gene_df, tss_peak, all_tss_peak

    def get_peaks_around_pos(self, celltype_id, chr_name, pos):
        """
        Get the peaks around a specific position with a maximum distance of MAX_CONTEXT_DISTANCE.
        """
        return self.datapool._query_peaks(
            celltype_id, chr_name, pos - self.MAX_CONTEXT_DISTANCE, pos + self.MAX_CONTEXT_DISTANCE).reset_index().rename(columns={'index': 'original_peak_index'}).reset_index()


class PerturbationInferenceDataset(Dataset):
    """
    Wrapper around InferenceDataset to allow for parallel processing for different mutations.

    Args:
        inference_dataset (InferenceDataset): The InferenceDataset to use for generating samples.
        perturbations (pandas.DataFrame): A pandas DataFrame containing the perturbations to apply.
        mode (str): The mode of perturbation to apply. Can be 'mutation' or 'peak_inactivation'.

    Returns:
    PerturbationInferenceDataset: A PerturbationInferenceDataset object.
        """

    def __init__(self, inference_dataset, perturbations, mode='mutation') -> None:
        super().__init__()
        self.inference_dataset = inference_dataset
        self.perturbations = perturbations
        self.gene_list = inference_dataset.accessible_genes
        self.gencode_obj = inference_dataset.gencode_obj
        self.mode = mode
        self.calculate_mutation_per_gene()
        print(f"n_celltype: {self.inference_dataset.datapool.n_celltypes}")
        print(f"n_gene: {len(self.gene_list)}")
        print(f"n_perturbation: {len(self.perturbations)}")
        print(f"overlapping perturbations across cell types: {len(self)}")

    def calculate_mutation_per_gene(self):
        """Not all mutations are in the same gene, calculate the number of mutations for each gene as a dictionary"""
        perturbations_gene_overlap = []
        for celltype, gene_list in self.gene_list.items():
            df = pr(self.perturbations).join(pr(self.gencode_obj.gtf).extend(
            2_000_000)).df.query('gene_name in @gene_list').drop(['index', 'Start_b', 'End_b', 'Strand'], axis=1)
            df['celltype'] = celltype
            perturbations_gene_overlap.append(df)
        self.perturbations_gene_overlap = pd.concat(
            perturbations_gene_overlap, ignore_index=True)
        

    def __len__(self):
        return len(self.perturbations_gene_overlap)

    def __getitem__(self, i):
        """return both wild type and mutated batch in a tuple. implement by calling the get_item_for_gene_in_celltype function with or without specify mutation or peak inactivation. loop through the mutations list or peak inactivation list."""
        celltype = self.perturbations_gene_overlap.celltype.values[i]
        gene_name = self.perturbations_gene_overlap.gene_name.values[i]
        perturbation = self.perturbations_gene_overlap.iloc[
            i:i+1]
        data_key = self.inference_dataset.datapool._get_data_key(
            celltype)
        args = {'mutations': perturbation, 'peak_inactivation': None} if self.mode == 'mutation' else {
            'mutations': None, 'peak_inactivation': perturbation}
        return {'WT': self.inference_dataset.get_item_for_gene_in_celltype(data_key, celltype, gene_name, mutations=None, peak_inactivation=None),
                'MUT': self.inference_dataset.get_item_for_gene_in_celltype(data_key, celltype, gene_name, **args)}


@dataclass
class ReferenceRegionMotifConfig:
    root: str = '/home/xf2217/Projects/get_data'
    data: str = 'fetal_tfatlas_peaks_motif.hg38.zarr'
    refdata: str = 'fetal_union_peak_motif_v1.hg38.zarr'
    motif_scaler: float = 1.0


class ReferenceRegionMotif(object):
    def __init__(self, cfg: ReferenceRegionMotifConfig) -> None:
        self.cfg = cfg
        self.dataset = zarr.open_group(
            os.path.join(cfg.root, cfg.data), mode='r')
        self.data = self.dataset['data'][:]
        self.peak_names = self.dataset['peak_names'][:]
        self.motif_names = self.dataset['motif_names'][:]
        self.refdataset = zarr.open_group(
            os.path.join(cfg.root, cfg.refdata), mode='r')
        self.refdata = self.refdataset['data'][:]
        self.refpeak_names = self.refdataset['peak_names'][:]
        self.motif_scaler = cfg.motif_scaler
        # reorder data to match sorted peak order
        self.data = self.data[self.peaks['index'].values]
        self.refdata = self.refdata[self.refpeaks['index'].values]
        # drop 'index' column in peaks
        self._peaks = self._peaks.drop('index', axis=1).reset_index()
        self._refpeaks = self._refpeaks.drop('index', axis=1).reset_index()

    @property
    def num_peaks(self):
        return len(self.peak_names)

    @property
    def num_motifs(self):
        return len(self.motif_names)

    @property
    def peaks(self):
        if not hasattr(self, '_peaks'):
            df = pd.DataFrame(self.peak_names[:])
            # split chr:start-end into chr, start, end
            df.columns = ['peak_names']
            df['Chromosome'] = df['peak_names'].apply(
                lambda x: x.split(':')[0])
            df['Start'] = df['peak_names'].apply(
                lambda x: x.split(':')[1].split('-')[0])
            df['End'] = df['peak_names'].apply(
                lambda x: x.split(':')[1].split('-')[1])
            df = pr(df.reset_index()).sort().df
            self._peaks = df
        return self._peaks

    @property
    def refpeaks(self):
        if not hasattr(self, '_refpeaks'):
            df = pd.DataFrame(self.refpeak_names[:])
            # split chr:start-end into chr, start, end
            df.columns = ['peak_names']
            df['Chromosome'] = df['peak_names'].apply(
                lambda x: x.split(':')[0])
            df['Start'] = df['peak_names'].apply(
                lambda x: x.split(':')[1].split('-')[0])
            df['End'] = df['peak_names'].apply(
                lambda x: x.split(':')[1].split('-')[1])
            df = pr(df.reset_index()).sort().df
            self._refpeaks = df
        return self._refpeaks

    @property
    def peak_names_to_index(self):
        return {name: i for i, name in enumerate(self.peak_names)}

    @staticmethod
    def get_cutoff(d):
        # for each column, find 90% quantile value and return a vector of cutoffs
        cutoffs = []
        for i in range(d.shape[1]):
            cutoffs.append(np.quantile(d[:, i], 0.9))
        return cutoffs

    def map_peaks_to_motifs(self, peaks, normalize=True):
        """
        Map peaks to motifs.

        Args:
            peak_names: List of peak names.
        """

        if isinstance(peaks, list) or isinstance(peaks, np.ndarray):
            peaks = self.peaks.query('peak_names.isin(@peaks)')
        elif isinstance(peaks, pd.DataFrame) and 'Chromosome' in peaks.columns and 'Start' in peaks.columns and 'End' in peaks.columns:
            if 'index' not in peaks.columns:
                peaks = peaks.reset_index()
            refpeaks = pr(self.refpeaks).join(
                pr(peaks).sort(), suffix='_input').df
            peaks = pr(self.peaks).join(pr(peaks).sort(), suffix='_input').df.query(
                'Start==Start_input & End==End_input').reset_index(drop=True)

        peak_indices = peaks['index'].values
        data = self.data[peak_indices]
        refpeak_indices = refpeaks['index'].values
        refdata = self.refdata[refpeak_indices]
        refdata_cutoff = self.get_cutoff(refdata)
        data = data * (data > refdata_cutoff)
        refdata = refdata * (refdata > refdata_cutoff)
        if normalize:
            data = data / refdata.max(0) / self.motif_scaler
            data[data > 1] = 1
        return data, peaks

    def __repr__(self) -> str:
        return f'ReferenceRegionMotif(num_peaks={self.num_peaks}, num_motifs={self.num_motifs})'


class ReferenceRegionDataset(Dataset):
    def __init__(self, reference_region_motif: ReferenceRegionMotif,
                 zarr_dataset: PretrainDataset,
                 transform=None,
                 quantitative_atac: bool = False,
                 sampling_step: int = 50,
                 ) -> None:
        super().__init__()
        self.reference_region_motif = reference_region_motif
        self.zarr_dataset = zarr_dataset
        self.transform = transform
        self.quantitative_atac = quantitative_atac
        self.sampling_step = sampling_step
        self.mask_ratio = zarr_dataset.mask_ratio
        self.is_train = zarr_dataset.is_train
        self.num_region_per_sample = zarr_dataset.n_peaks_upper_bound
        self.leave_out_celltypes = zarr_dataset.leave_out_celltypes
        self.leave_out_chromosomes = zarr_dataset.leave_out_chromosomes
        self.peak_count_filter = zarr_dataset.peak_count_filter

        self.peak_names = reference_region_motif.peak_names
        self.setup()
        self.data_dict

    @property
    def data_dict(self):
        if not hasattr(self, '_data_dict'):
            self._data_dict = {data_key: self.reference_region_motif.map_peaks_to_motifs(
                peaks) for data_key, peaks in self.zarr_dataset.datapool.peaks_dict.items()}
        return self._data_dict

    def extract_data_list(self, region_motif, peaks):
        region_motif_list = []
        peak_list = []
        target_list = []
        tssidx_list = []
        hic_matrix_list = []

        all_chromosomes = peaks["Chromosome"].unique(
        ).tolist()
        input_chromosomes = _chromosome_splitter(
            all_chromosomes, self.leave_out_chromosomes, is_train=self.is_train
        )
        target_data = peaks[["Expression_positive",
                             "Expression_negative"]].values
        tssidx_data = peaks["TSS"].values
        atpm = peaks['aTPM'].values
        target_data[atpm < 0.05, :] = 0
        if not self.quantitative_atac:
            region_motif = np.concatenate(
                [region_motif, np.zeros((region_motif.shape[0], 1))+1], axis=1)
        else:
            region_motif = np.concatenate(
                [region_motif, atpm.reshape(-1, 1)/atpm.reshape(-1, 1).max()], axis=1)
        for chromosome in input_chromosomes:
            idx_peak_list = peaks.query('Chromosome==@chromosome').index.values
            idx_peak_start = idx_peak_list[0]
            idx_peak_end = idx_peak_list[-1]
            for i in range(idx_peak_start, idx_peak_end, self.sampling_step):
                shift = np.random.randint(-self.sampling_step //
                                          2, self.sampling_step // 2)
                start_index = max(0, i + shift)
                end_index = start_index + self.num_region_per_sample

                celltype_peak_annot_i = peaks.iloc[start_index:end_index, :]
                if celltype_peak_annot_i.shape[0] == 0:
                    continue
                if celltype_peak_annot_i.iloc[-1].End - celltype_peak_annot_i.iloc[0].Start > 5000000:
                    end_index = celltype_peak_annot_i[celltype_peak_annot_i.End -
                                                      celltype_peak_annot_i.Start < 5000000].index[-1]
                if celltype_peak_annot_i["Start"].min() < 0 or celltype_peak_annot_i.shape[0] != self.num_region_per_sample:
                    continue

                region_motif_i = coo_matrix(
                    region_motif[start_index:end_index])

                target_i = coo_matrix(
                    target_data[start_index:end_index])
                tssidx_i = tssidx_data[start_index:end_index]

                if region_motif_i.shape[0] == self.num_region_per_sample:

                    # get hic matrix for celltype_peak_annot_i
                    if self.zarr_dataset.datapool.hic_obj is not None:
                        hic_matrix_i = get_hic_from_idx(
                            self.zarr_dataset.datapool.hic_obj, celltype_peak_annot_i)
                        if hic_matrix_i is None:
                            continue
                        else:
                            hic_matrix_list.append(hic_matrix_i)
                    else:
                        hic_matrix_list.append(0)

                    region_motif_list.append(region_motif_i)
                    peak_list.append(celltype_peak_annot_i)
                    target_list.append(target_i)
                    tssidx_list.append(tssidx_i)
        return peak_list, region_motif_list, target_list, tssidx_list, hic_matrix_list

    def setup(self):
        self.sample_indices = []
        
        for data_key, (region_motif, peaks) in tqdm(self.data_dict.items()):
            all_chromosomes = peaks["Chromosome"].unique().tolist()
            input_chromosomes = _chromosome_splitter(
                all_chromosomes, self.leave_out_chromosomes, is_train=self.is_train
            )
            
            for chromosome in input_chromosomes:
                idx_peak_list = peaks.query('Chromosome==@chromosome').index.values
                idx_peak_start = idx_peak_list[0]
                idx_peak_end = idx_peak_list[-1]
                
                for i in range(idx_peak_start, idx_peak_end, self.sampling_step):
                    shift = np.random.randint(-self.sampling_step // 2, self.sampling_step // 2)
                    start_index = max(0, i + shift)
                    end_index = start_index + self.num_region_per_sample
                    
                    if end_index <= region_motif.shape[0]:
                        self.sample_indices.append((data_key, start_index, end_index))
                        
    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        data_key, start_index, end_index = self.sample_indices[index]
        region_motif, peaks = self.data_dict[data_key]
        
        region_motif_i = region_motif[start_index:end_index]
        peak_i = peaks.iloc[start_index:end_index]
        
        target_data = peaks[["Expression_positive", "Expression_negative"]].values
        tssidx_data = peaks["TSS"].values
        atpm = peaks['aTPM'].values
        target_data[atpm < 0.05, :] = 0
        
        if not self.quantitative_atac:
            region_motif_i = np.concatenate(
                [region_motif_i, np.zeros((region_motif_i.shape[0], 1))+1], axis=1)
        else:
            region_motif_i = np.concatenate(
                [region_motif_i, atpm[start_index:end_index].reshape(-1, 1)/atpm[start_index:end_index].reshape(-1, 1).max()], axis=1)
        
        target_i = coo_matrix(target_data[start_index:end_index])
        tssidx_i = tssidx_data[start_index:end_index]
        
        if self.zarr_dataset.datapool.hic_obj is not None:
            hic_matrix_i = get_hic_from_idx(self.zarr_dataset.datapool.hic_obj, peak_i)
            if hic_matrix_i is None:
                # return a matrix with all zeros
                hic_matrix_i = np.zeros((self.num_region_per_sample, self.num_region_per_sample))
        else:
            hic_matrix_i = 0
        
        if self.mask_ratio > 0:
            mask = np.hstack(
                [
                    np.zeros(int(self.num_region_per_sample - self.num_region_per_sample*self.mask_ratio)),
                    np.ones(int(self.num_region_per_sample*self.mask_ratio)),
                ]
            )
            np.random.shuffle(mask)
        else:
            mask = tssidx_i
        
        if self.transform is not None:
            region_motif_i, mask, target_i = self.transform(region_motif_i, tssidx_i, target_i)
        
        if region_motif_i.shape[0] == 1:
            region_motif_i = region_motif_i.squeeze(0)
        
        data = {'region_motif': region_motif_i.astype(np.float32),
                'mask': mask,
                'chromosome': peak_i['Chromosome'].values[0],
                'peak_coord': peak_i[['Start', 'End']].values,
                'exp_label': target_i.toarray().astype(np.float32),
                'hic_matrix': hic_matrix_i}

        return data

class InferenceReferenceRegionDataset(Dataset):
    def __init__(self, reference_region_motif: ReferenceRegionMotif,
                 zarr_dataset: InferenceDataset,
                 quantitative_atac: bool = False,
                 sampling_step: int = 50,
                 ) -> None:
        super().__init__()

        self.mask_ratio = zarr_dataset.mask_ratio
        self.is_train = zarr_dataset.is_train
        self.num_region_per_sample = zarr_dataset.n_peaks_upper_bound
        self.leave_out_celltypes = zarr_dataset.leave_out_celltypes
        self.leave_out_chromosomes = zarr_dataset.leave_out_chromosomes
        self.quantitative_atac = quantitative_atac
        self.reference_region_motif = reference_region_motif
        self.peak_count_filter = zarr_dataset.peak_count_filter
        self.peak_names = reference_region_motif.peak_names

        self.zarr_dataset = zarr_dataset
        self.data_dict

    @property
    def data_dict(self):
        if not hasattr(self, '_data_dict'):
            self._data_dict = {data_key: self.reference_region_motif.map_peaks_to_motifs(
                peaks) for data_key, peaks in self.zarr_dataset.datapool.peaks_dict.items()}
        return self._data_dict

    def __len__(self):
        return len(self.zarr_dataset)

    def __getitem__(self, index):
        sample = self.zarr_dataset[index]
        target = sample['additional_peak_features'][:, 0:2]
        tssidx = sample['additional_peak_features'][:, 3]
        hic_matrix = sample['hic_matrix']
        mask = tssidx
        peak_start = sample['metadata']['original_peak_start']
        celltype_id = sample['metadata']['celltype_id']
        strand = sample['metadata']['strand']
        gene_name = sample['metadata']['gene_name']
        peak_end = peak_start + target.shape[0]
        region_motif, peaks = self.data_dict[celltype_id]
        region_motif = region_motif[peak_start:peak_end]
        chromosome = peaks['Chromosome'].values[peak_start]
        peak_coord = peaks[['Start', 'End']].values[peak_start:peak_end]
        atpm = peaks['aTPM'].values[peak_start:peak_end]
        # append binary or quantitative atac signal
        if not self.quantitative_atac:
            region_motif = np.concatenate(
                [region_motif, np.zeros((region_motif.shape[0], 1))+1], axis=1)
        else:
            region_motif = np.concatenate(
                [region_motif, atpm.reshape(-1, 1)/atpm.reshape(-1, 1).max()], axis=1)

        # right zero padding
        pad_length = self.num_region_per_sample - region_motif.shape[0]
        max_tss_count = 100

        region_motif = np.pad(region_motif, ((0, pad_length), (0, 0)), mode='constant')
        peak_coord = np.pad(peak_coord, ((0, pad_length), (0, 0)), mode='constant')
        mask = np.pad(mask, (0, pad_length), mode='constant')
        exp_label = np.pad(target, ((0, pad_length), (0, 0)), mode='constant')
        tss_peak = sample['metadata']['tss_peak']
        if tss_peak.shape == ():
            tss_peak = np.array([tss_peak])
        all_tss_peak = np.pad(sample['metadata']['all_tss_peak'], (0, max_tss_count - len(sample['metadata']['all_tss_peak'])), mode='constant', constant_values=-1)

        return {'region_motif': region_motif.astype(np.float32),
                'chromosome': chromosome,
                'peak_coord': peak_coord,
                'mask': mask,
                'exp_label': exp_label.astype(np.float32),
                'hic_matrix': hic_matrix,
                'strand': strand,
                'gene_name': gene_name,
                'tss_peak': tss_peak,
                'all_tss_peak': all_tss_peak,
        }

class PerturbationInferenceReferenceRegionDataset(Dataset):
    """
    Wrapper around InferenceReferenceRegionDataset to allow for parallel processing for different mutations.

    Args:
        inference_dataset (InferenceReferenceRegionDataset): The InferenceReferenceRegionDataset to use for generating samples.
        perturbations (pandas.DataFrame): A pandas DataFrame containing the perturbations to apply.
        mode (str): The mode of perturbation to apply. Can be 'mutation' or 'peak_inactivation'.
    """

    def __init__(self, inference_dataset, perturbations, mode='mutation') -> None:
        super().__init__()
        self.inference_dataset = inference_dataset
        self.perturbations = perturbations
        self.gene_list = inference_dataset.zarr_dataset.accessible_genes #{celltype_id: gene_list_for_celltype}
        self.gencode_obj = inference_dataset.zarr_dataset.gencode_obj
        self.mode = mode
        self._calculate_mutation_per_gene()
        print(
            f"n_celltype: {self.inference_dataset.zarr_dataset.datapool.n_celltypes}")
        print(f"n_gene: {len(self.gene_list)}")
        print(f"n_perturbation: {len(self.perturbations)}")

    def _calculate_mutation_per_gene(self):
        """Not all mutations are in the same gene, calculate the number of mutations for each gene as a dictionary"""
        from pyranges import PyRanges as pr
        if 'gene_name' not in self.perturbations.columns:
            perturbations_gene_overlap = []
            for celltype, gene_list in self.gene_list.items():
                # Extend TSS to 4mbp
                celltype_perturbation = pr(self.perturbations).join(
                    pr(self.gencode_obj.gtf).extend(2_000_000)
                ).df.query('gene_name in @gene_list').drop(
                    ['index', 'Start_b', 'End_b', 'Strand'], axis=1
                )
                celltype_perturbation['celltype'] = celltype
                perturbations_gene_overlap.append(celltype_perturbation)

            self.perturbations_gene_overlap = pd.concat(
                perturbations_gene_overlap)
            # if empty, raise error
            if self.perturbations_gene_overlap.shape[0] == 0:
                raise ValueError(
                    "No perturbations found in the gene list, please check the gene list and perturbations."
                )
        else:
            perturbations_gene_overlap = []
            for celltype, gene_list in self.gene_list.items():
                celltype_perturbation = self.perturbations.query('gene_name in @gene_list') 
                celltype_perturbation['celltype'] = celltype
                perturbations_gene_overlap.append(celltype_perturbation)

            self.perturbations_gene_overlap = pd.concat(perturbations_gene_overlap)
            if self.perturbations_gene_overlap.shape[0] == 0:
                raise ValueError(
                    "No perturbations found in the gene list, please check the gene list and perturbations."
                )

    def __len__(self):
        return len(self.perturbations_gene_overlap)

    def __getitem__(self, i):
        celltype = self.perturbations_gene_overlap.celltype.values[i]
        gene_name = self.perturbations_gene_overlap.gene_name.values[i]
        perturbation = self.perturbations_gene_overlap.iloc[
            i:i + 1
        ]
        data_key = self.inference_dataset.zarr_dataset.datapool._get_data_key(
            celltype)

        wt_sample = self._get_sample(data_key, celltype, gene_name)
        mut_sample = self._get_sample(
            data_key, celltype, gene_name, perturbation)

        return {'WT': wt_sample, 'MUT': mut_sample}

    def _get_sample(self, data_key, celltype_id, gene_name, perturbation=None):
        sample = self.inference_dataset.zarr_dataset.get_item_for_gene_in_celltype(
            data_key, celltype_id, gene_name,
            mutations=perturbation if self.mode == 'mutation' else None,
            peak_inactivation=perturbation if self.mode == 'peak_inactivation' else None
        )
        target = sample['additional_peak_features'][:, 0:2]
        tssidx = sample['additional_peak_features'][:, 3]
        hic_matrix = sample['hic_matrix']
        mask = tssidx

        peak_start = sample['metadata']['original_peak_start']
        peak_end = peak_start + target.shape[0]

        region_motif = self.inference_dataset.data_dict[celltype_id][0][peak_start:peak_end]
        peaks = self.inference_dataset.data_dict[celltype_id][1][peak_start:peak_end]
        atpm = peaks['aTPM'].values

        if self.mode == 'peak_inactivation' and perturbation is not None:
            region_motif, atpm = self._apply_peak_inactivation(
                region_motif, perturbation, peaks, atpm)

        if not self.inference_dataset.quantitative_atac:
            region_motif = np.concatenate(
                [region_motif, np.zeros((region_motif.shape[0], 1)) + 1], axis=1
            )
        else:
            region_motif = np.concatenate(
                [region_motif, atpm.reshape(-1, 1) / atpm.reshape(-1, 1).max()], axis=1
            )



        return {
            'region_motif': region_motif.astype(np.float32),
            'mask': mask,
            'exp_label': target.astype(np.float32),
            'hic_matrix': hic_matrix,
            'perturb_chrom': perturbation.Chromosome.values[0] if perturbation is not None else 0,
            'perturb_start': perturbation.Start.values[0] if perturbation is not None else 0,
            'perturb_end': perturbation.End.values[0] if perturbation is not None else 0,
            'strand': sample['metadata']['strand'],
            'gene_name': sample['metadata']['gene_name'],
            'tss_peak': sample['metadata']['tss_peak'],
            'all_tss_peak': sample['metadata']['all_tss_peak']
        }

    def _apply_peak_inactivation(self, region_motif, perturbation, peaks, atpm):
        """Apply peak inactivation by setting the corresponding peak/region in region_motif to 0."""
        perturbed_region_motif = region_motif.copy()
        peaks_df = pr(peaks[['Chromosome', 'Start', 'End']].reset_index(drop=True).reset_index())
        perturbation_df = pr(perturbation)

        overlap = peaks_df.join(perturbation_df, suffix='_perturb').df
        try:
            overlap_indices = overlap['index'].values
            overlap_indices = overlap_indices[overlap_indices <
                                              region_motif.shape[0]]
            overlap_indices = overlap_indices[overlap_indices > 0]
            perturbed_region_motif[overlap_indices] = 0
            atpm[overlap_indices] = 0
            return perturbed_region_motif, atpm
        except:
            logging.warning(
                f"Failed to apply peak inactivation for {perturbation}, no overlapping peaks found.")
            return region_motif, atpm


class RegionDataset(Dataset):
    """
    PyTorch dataset class for cell type expression data.

    Args:
        root (str): Root directory path.
        num_region_per_sample (int): Number of regions for each sample.
        is_train (bool, optional): Specify if the dataset is for training. Defaults to True.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        args (Any, optional): Additional arguments. Defaults to None.

    Attributes:
        root (str): Root directory path.
        transform (callable): Transform function.
        peaks (List[coo_matrix]): List of peak data.
        targets (List[np.ndarray]): List of target data.
        tssidxs (np.ndarray): Array of TSS indices.
    """

    def __init__(
        self,
        root: str,
        metadata_path: str,
        num_region_per_sample: int,
        transform: Optional[Callable] = None,
        data_type: str = "fetal",
        is_train: bool = True,
        leave_out_celltypes: str = "",
        leave_out_chromosomes: str = "",
        quantitative_atac: bool = False,
        sampling_step: int = 100,
        mask_ratio: float = 0.0,
    ) -> None:
        super().__init__()

        self.root = root
        self.transform = transform
        self.is_train = is_train
        self.leave_out_celltypes = leave_out_celltypes
        self.leave_out_chromosomes = leave_out_chromosomes
        self.quantitative_atac = quantitative_atac
        self.sampling_step = sampling_step
        self.num_region_per_sample = num_region_per_sample
        self.mask_ratio = mask_ratio
        metadata_path = os.path.join(
            self.root, metadata_path
        )
        peaks, targets, tssidx = self._make_dataset(
            False,
            data_type,
            self.root,
            metadata_path,
            num_region_per_sample,
            leave_out_celltypes,
            leave_out_chromosomes,
            quantitative_atac,
            is_train,
            sampling_step,
        )

        if len(peaks) == 0:
            raise RuntimeError(f"Found 0 files in subfolders of: {self.root}")

        self.peaks = peaks
        self.targets = targets
        self.tssidxs = np.array(tssidx)

    def __repr__(self) -> str:
        return f"""
Total {'train' if self.is_train else 'test'} samples: {len(self.peaks)}
Leave out celltypes: {self.leave_out_celltypes}
Leave out chromosomes: {self.leave_out_chromosomes}
Use quantitative_atac: {self.quantitative_atac}
Sampling step: {self.sampling_step}
        """

    def __getitem__(self, index: int) -> Tuple[coo_matrix, np.ndarray, np.ndarray]:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            Tuple[coo_matrix, np.ndarray, np.ndarray]: Tuple containing peak data, mask, and target data.
        """
        peak = self.peaks[index]
        target = self.targets[index]
        tssidx = self.tssidxs[index]
        if self.mask_ratio > 0:
            mask = np.hstack(
                [
                    np.zeros(int(self.num_region_per_sample -
                                 self.num_region_per_sample*self.mask_ratio)),
                    np.ones(int(self.num_region_per_sample*self.mask_ratio)),
                ]
            )
            np.random.shuffle(mask)
        else:
            mask = tssidx
        if self.transform is not None:
            peak, mask, target = self.transform(peak, tssidx, target)
        if peak.shape[0] == 1:
            peak = peak.squeeze(0)
        return {'region_motif': peak.toarray().astype(np.float32),
                'mask': mask,
                'exp_label': target.toarray().astype(np.float32)}

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.peaks)

    @staticmethod
    def _cell_splitter(
        celltype_metadata: pd.DataFrame,
        leave_out_celltypes: str,
        datatypes: str,
        is_train: bool = True,
        is_pretrain: bool = False,
    ) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
        """
        Process data based on given parameters.

        Args:
            celltype_metadata (pd.DataFrame): Cell type metadata dataframe.
            leave_out_celltypes (str): Comma-separated string of cell types to be excluded or used for validation.
            datatypes (str): Comma-separated string of data types to be considered.
            is_train (bool, optional): Specify if the processing is for training data. Defaults to True.
            is_pretrain (bool, optional): Specify if the processing is for pre-training data. Defaults to False.

        Returns:
            Tuple[List[str], Dict[str, str], Dict[str, str]]: Tuple containing the list of target file IDs,
            cell labels dictionary, and datatype dictionary.
        """
        leave_out_celltypes = leave_out_celltypes.split(",")
        datatypes = datatypes.split(",")

        celltype_list = sorted(celltype_metadata["celltype"].unique().tolist())
        if is_train:
            # TODO: revert this
            # celltype_list = [
            #     cell for cell in celltype_list if cell not in leave_out_celltypes]
            # print(f"Train cell types list: {celltype_list}")
            # print(f"Train data types list: {datatypes}")

            celltype_list = leave_out_celltypes if leave_out_celltypes != [
                ""] else celltype_list
            print(
                f"Using validation cell type for training!!! cell types list: {celltype_list}")
            print(
                f"Using validation cell type for training!!! data types list: {datatypes}")

        else:
            celltype_list = leave_out_celltypes if leave_out_celltypes != [
                ""] else celltype_list
            print(f"Validation cell types list: {celltype_list}")
            print(f"Validation data types list: {datatypes}")

        file_id_list = []
        datatype_dict = {}
        cell_dict = {}
        for cell in celltype_list:
            celltype_metadata_of_cell = celltype_metadata[celltype_metadata["celltype"] == cell]
            for file, cluster, datatype, expression in zip(
                celltype_metadata_of_cell["id"],
                celltype_metadata_of_cell["cluster"],
                celltype_metadata_of_cell["datatype"],
                celltype_metadata_of_cell["expression"],
            ):
                if is_pretrain and datatype in datatypes:
                    file_id_list.append(file)
                    cell_dict[file] = cluster
                    datatype_dict[file] = datatype
                elif datatype in datatypes and expression == "True":
                    file_id_list.append(file)
                    cell_dict[file] = cluster
                    datatype_dict[file] = datatype

        if not is_train:
            file_id_list = sorted(file_id_list)

        print(f"File ID list: {file_id_list}")
        return file_id_list, cell_dict, datatype_dict

    @staticmethod
    def _generate_paths(file_id: int, data_path: str, data_type: str, quantitative_atac: bool = False) -> dict:
        """
        Generate a dictionary of paths based on the given parameters.

        Args:
            file_id (int): File ID.
            data_path (str): Path to the data directory.
            data_type (str): Data type.
            quantitative_atac (bool, optional): Specify if quantitative atac files should be used. Defaults to False.

        Returns:
            dict: Dictionary of paths with file IDs as keys and corresponding paths as values.

        Raises:
            FileNotFoundError: If the peak file is not found.
        """
        peak_npz_path = os.path.join(
            data_path, data_type, f"{file_id}.watac.npz"
        )

        if not os.path.exists(peak_npz_path):
            raise FileNotFoundError(f"Peak file not found: {peak_npz_path}")

        target_npy_path = os.path.join(
            data_path, data_type, f"{file_id}.exp.npy")
        tssidx_npy_path = os.path.join(
            data_path, data_type, f"{file_id}.tss.npy")
        celltype_annot = os.path.join(
            data_path, data_type, f"{file_id}.csv")
        # if celltype_annot is not exist, check csv.gz
        if not os.path.exists(celltype_annot):
            celltype_annot = os.path.join(
                data_path, data_type, f"{file_id}.csv.gz")
        exp_feather = os.path.join(
            data_path, data_type, f"{file_id}.exp.feather"
        )
        return {
            "file_id": file_id,
            "peak_npz": peak_npz_path,
            "target_npy": target_npy_path,
            "tssidx_npy": tssidx_npy_path,
            "celltype_annot_csv": celltype_annot,
            "exp_feather": exp_feather,
        }

    def _make_dataset(
        self,
        is_pretrain: bool,
        datatypes: str,
        data_path: str,
        celltype_metadata_path: str,
        num_region_per_sample: int,
        leave_out_celltypes: str,
        leave_out_chromosomes: str,
        quantitative_atac: bool,
        is_train: bool,
        step: int = 200,
    ) -> Tuple[List[coo_matrix], List[coo_matrix], List[np.ndarray]]:
        """
        Generate a dataset for training or testing.

        Args:
            is_pretrain (bool): Whether it is a pretraining dataset.
            datatypes (str): String of comma-separated data types.
            data_path (str): Path to the data.
            celltype_metadata_path (str): Path to the celltype metadata file.
            num_region_per_sample (int): Number of regions per sample.
            leave_out_celltypes (str): String of comma-separated cell types to leave out.
            leave_out_chromosomes (str): String of comma-separated chromosomes to leave out.
            quantitative_atac (bool): Whether to use peak data with no ATAC count values.
            is_train (bool): Whether it is a training dataset.
            step (int, optional): Step size for generating samples. Defaults to 200.

        Returns:
            Tuple[List[coo_matrix], List[str], List[coo_matrix], List[np.ndarray]]: Tuple containing the generated peak data,
            cell labels, target data, and TSS indices.
        """
        celltype_metadata = pd.read_csv(
            celltype_metadata_path, sep=",", dtype=str)
        file_id_list, cell_dict, datatype_dict = self._cell_splitter(
            celltype_metadata,
            leave_out_celltypes,
            datatypes,
            is_train=is_train,
            is_pretrain=is_pretrain,
        )
        peak_list = []
        cell_list = []
        target_list = [] if not is_pretrain else None
        tssidx_list = [] if not is_pretrain else None

        for file_id in file_id_list:
            cell_label = cell_dict[file_id]
            data_type = datatype_dict[file_id]
            print(file_id, data_path, data_type)
            paths_dict = self._generate_paths(
                file_id, data_path, data_type, quantitative_atac=quantitative_atac
            )

            celltype_peak_annot = pd.read_csv(
                paths_dict["celltype_annot_csv"], sep=",")

            try:
                peak_data = load_npz(paths_dict["peak_npz"])
                print(f"Feature shape: {peak_data.shape}")
            except FileNotFoundError:
                print(f"File not found - FILE ID: {file_id}")
                continue

            if not is_pretrain:
                target_data = np.load(paths_dict["target_npy"])
                tssidx_data = np.load(paths_dict["tssidx_npy"])
                print(f"Target shape: {target_data.shape}")
                atac_cutoff = 1 - \
                    (peak_data[:, 282] >= 0.05).toarray().flatten()
                target_data[atac_cutoff, :] = 0

            if quantitative_atac is False:
                peak_data[:, 282] = 1

            all_chromosomes = celltype_peak_annot["Chromosome"].unique(
            ).tolist()
            input_chromosomes = _chromosome_splitter(
                all_chromosomes, leave_out_chromosomes, is_train=is_train
            )

            for chromosome in input_chromosomes:
                idx_peak_list = celltype_peak_annot.index[celltype_peak_annot["Chromosome"] == chromosome].tolist(
                )
                idx_peak_start = idx_peak_list[0]
                idx_peak_end = idx_peak_list[-1]
                for i in range(idx_peak_start, idx_peak_end, step):
                    # shift = np.random.randint(-step // 2, step // 2)
                    start_index = i  # max(0, i + shift)
                    end_index = start_index + num_region_per_sample

                    celltype_annot_i = celltype_peak_annot.iloc[start_index:end_index, :]
                    if celltype_annot_i.shape[0] < num_region_per_sample:
                        continue
                    # if celltype_annot_i.iloc[-1].End - celltype_annot_i.iloc[0].Start > 5000000:
                    #     end_index = celltype_annot_i[celltype_annot_i.End -
                    #                                  celltype_annot_i.Start < 5000000].index[-1]
                    if celltype_annot_i["Start"].min() < 0:
                        continue
                    peak_data_i = coo_matrix(peak_data[start_index:end_index])

                    if not is_pretrain:
                        target_i = coo_matrix(
                            target_data[start_index:end_index])
                        tssidx_i = tssidx_data[start_index:end_index]

                    if peak_data_i.shape[0] == num_region_per_sample:
                        peak_list.append(peak_data_i)
                        cell_list.append(cell_label)
                        if not is_pretrain:
                            target_list.append(target_i)
                            tssidx_list.append(tssidx_i)

        return peak_list, target_list, tssidx_list


class InferenceRegionDataset(RegionDataset):
    """Same as RegionDataset but load the exp.feather to get gene index in peaks

    Args:
        root (str): Root directory path.
        metadata_path (str): Path to the metadata file.
        num_region_per_sample (int): Number of regions for each sample.
        transform (Optional[Callable], optional): Transform function. Defaults to None.
        data_type (str, optional): Data type. Defaults to "fetal".
        is_train (bool, optional): Specify if the dataset is for training. Defaults to True.
        leave_out_celltypes (str, optional): Comma-separated string of cell types to leave out. Defaults to "".
        leave_out_chromosomes (str, optional): Comma-separated string of chromosomes to leave out. Defaults to "".
        quantitative_atac (bool, optional): Specify if quantitative ATAC data should be used. Defaults to False.
        sampling_step (int, optional): Sampling step. Defaults to 100.
        mask_ratio (float, optional): Mask ratio. Defaults to 0.0.
        gene_list ([type], optional): Gene list. Defaults to None.
        gencode_obj ([type], optional): Gencode object. Defaults to None.
    """

    def __init__(
        self,
        root: str,
        metadata_path: str,
        num_region_per_sample: int,
        transform: Optional[Callable] = None,
        data_type: str = "fetal",
        is_train: bool = True,
        leave_out_celltypes: str = "",
        leave_out_chromosomes: str = "",
        quantitative_atac: bool = False,
        sampling_step: int = 100,
        mask_ratio: float = 0.0,
        gene_list=None,
        gencode_obj=None,

    ) -> None:
        self.root = root
        self.transform = transform
        self.is_train = is_train
        self.leave_out_celltypes = leave_out_celltypes
        self.leave_out_chromosomes = leave_out_chromosomes
        self.quantitative_atac = quantitative_atac
        self.sampling_step = sampling_step
        self.num_region_per_sample = num_region_per_sample
        self.mask_ratio = mask_ratio
        metadata_path = os.path.join(
            self.root, metadata_path
        )
        if isinstance(gene_list, str):
            if ',' in gene_list:
                gene_list = gene_list.split(',')
            elif os.path.exists(gene_list):
                gene_list = np.loadtxt(gene_list, dtype=str)
        self.gene_list = gene_list if gene_list is not None else []
        self.gencode_obj = gencode_obj
        peaks, targets, tssidx, gene_names, strands, tss_peaks = self._make_dataset(
            False,
            data_type,
            self.root,
            metadata_path,
            num_region_per_sample,
            leave_out_celltypes,
            leave_out_chromosomes,
            quantitative_atac,
            is_train,
            sampling_step,
        )

        if len(peaks) == 0:
            raise RuntimeError(f"Found 0 files in subfolders of: {self.root}")

        self.peaks = peaks
        self.targets = targets
        self.tssidxs = np.array(tssidx)
        self.gene_names = gene_names
        self.strands = strands
        self.tss_peaks = tss_peaks

    def _make_dataset(
        self,
        is_pretrain: bool,
        datatypes: str,
        data_path: str,
        celltype_metadata_path: str,
        num_region_per_sample: int,
        leave_out_celltypes: str,
        leave_out_chromosomes: str,
        quantitative_atac: bool,
        is_train: bool,
        step: int = 200,
    ) -> Tuple[List[coo_matrix], List[coo_matrix], List[np.ndarray]]:
        """
        Generate a dataset for training or testing.

        Args:
            is_pretrain (bool): Whether it is a pretraining dataset.
            datatypes (str): String of comma-separated data types.
            data_path (str): Path to the data.
            celltype_metadata_path (str): Path to the celltype metadata file.
            num_region_per_sample (int): Number of regions per sample.
            leave_out_celltypes (str): String of comma-separated cell types to leave out.
            leave_out_chromosomes (str): String of comma-separated chromosomes to leave out.
            quantitative_atac (bool): Whether to use peak data with no ATAC count values.
            is_train (bool): Whether it is a training dataset.
            step (int, optional): Step size for generating samples. Defaults to 200.

        Returns:
            Tuple[List[coo_matrix], List[str], List[coo_matrix], List[np.ndarray]]: Tuple containing the generated peak data,
            cell labels, target data, and TSS indices.
        """
        celltype_metadata = pd.read_csv(
            celltype_metadata_path, sep=",", dtype=str)
        file_id_list, cell_dict, datatype_dict = self._cell_splitter(
            celltype_metadata,
            leave_out_celltypes,
            datatypes,
            is_train=is_train,
            is_pretrain=is_pretrain,
        )
        peak_list = []
        cell_list = []
        target_list = [] if not is_pretrain else None
        tssidx_list = [] if not is_pretrain else None
        gene_list = []
        strand_list = []
        tss_peak_list = []
        for file_id in file_id_list:
            cell_label = cell_dict[file_id]
            data_type = datatype_dict[file_id]
            print(file_id, data_path, data_type)
            paths_dict = self._generate_paths(
                file_id, data_path, data_type, quantitative_atac=quantitative_atac
            )

            celltype_peak_annot = pd.read_csv(
                paths_dict["celltype_annot_csv"], sep=",")

            try:
                peak_data = load_npz(paths_dict["peak_npz"])
                print(f"Feature shape: {peak_data.shape}")
            except FileNotFoundError:
                print(f"File not found - FILE ID: {file_id}")
                continue

            if os.path.exists(paths_dict["exp_feather"]):
                exp_df = pd.read_feather(paths_dict["exp_feather"])
            else:
                # construct exp_df from gencode_obj and save it to feather
                exp_df = self.gencode_obj.get_exp_feather(
                    celltype_peak_annot.drop('index', axis=1).reset_index())
                exp_df.to_feather(paths_dict["exp_feather"])

            if not is_pretrain:
                target_data = np.load(paths_dict["target_npy"])
                tssidx_data = np.load(paths_dict["tssidx_npy"])
                print(f"Target shape: {target_data.shape}")
                atac_cutoff = 1 - \
                    (peak_data[:, 282] >= 0.05).toarray().flatten()
                target_data[atac_cutoff, :] = 0

            if quantitative_atac is False:
                peak_data[:, 282] = 1

            all_chromosomes = celltype_peak_annot["Chromosome"].unique(
            ).tolist()
            input_chromosomes = _chromosome_splitter(
                all_chromosomes, leave_out_chromosomes, is_train=is_train
            )

            exp_df = exp_df.query(
                'gene_name.isin(@self.gene_list) & Chromosome.isin(@input_chromosomes)')

            # instead of loop over chromosome, loop over gene
            for gene, gene_df in exp_df.groupby('gene_name'):
                gene_name = gene_df['gene_name'].values[0]
                tss_peak = gene_df['index'].values
                strand = 0 if gene_df['Strand'].values[0] == '+' else 1
                idx = gene_df['index'].values[0] if strand == 0 else gene_df['index'].values[-1]

                start_idx = idx-num_region_per_sample//2
                end_idx = idx+num_region_per_sample//2
                if start_idx < 0 or end_idx >= peak_data.shape[0]:
                    continue
                celltype_annot_i = celltype_peak_annot.iloc[start_idx:end_idx, :]
                if celltype_annot_i.shape[0] < num_region_per_sample:
                    continue
                peak_data_i = coo_matrix(peak_data[start_idx:end_idx])

                if not is_pretrain:
                    target_i = coo_matrix(
                        target_data[start_idx:end_idx])
                    tssidx_i = tssidx_data[start_idx:end_idx]

                if peak_data_i.shape[0] == num_region_per_sample:
                    peak_list.append(peak_data_i)
                    cell_list.append(cell_label)
                    if not is_pretrain:
                        target_list.append(target_i)
                        tssidx_list.append(tssidx_i)
                    gene_list.append(gene_name)
                    strand_list.append(strand)
                    tss_peak = tss_peak-start_idx
                    tss_peak = tss_peak[tss_peak < num_region_per_sample]
                    tss_peak_list.append(tss_peak)

        return peak_list, target_list, tssidx_list, gene_list, strand_list, tss_peak_list

    def __getitem__(self, index: int):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            Tuple[coo_matrix, np.ndarray, np.ndarray]: Tuple containing peak data, mask, and target data.
        """
        peak = self.peaks[index]
        target = self.targets[index]
        tssidx = self.tssidxs[index]
        gene_name = self.gene_names[index]
        strand = self.strands[index]
        tss_peak = self.tss_peaks[index]
        tss_peak_mask = np.zeros(self.num_region_per_sample)
        tss_peak_mask[tss_peak] = 1
        if self.mask_ratio > 0:
            mask = np.hstack(
                [
                    np.zeros(int(self.num_region_per_sample -
                                 self.num_region_per_sample*self.mask_ratio)),
                    np.ones(int(self.num_region_per_sample*self.mask_ratio)),
                ]
            )
            np.random.shuffle(mask)
        else:
            mask = tssidx
        if self.transform is not None:
            peak, mask, target = self.transform(peak, tssidx, target)
        if peak.shape[0] == 1:
            peak = peak.squeeze(0)
        return {'region_motif': peak.toarray().astype(np.float32),
                'mask': mask,
                'gene_name': gene_name,
                'tss_peak': tss_peak_mask,
                'strand': strand,
                'exp_label': target.toarray().astype(np.float32)}

class EverythingDataset(ReferenceRegionDataset):
    def __init__(self, reference_region_motif: ReferenceRegionMotif,
                 zarr_dataset: PretrainDataset,
                 transform=None,
                 quantitative_atac: bool = False,
                 sampling_step: int = 50,
                 ) -> None:
        super().__init__(reference_region_motif, zarr_dataset, transform, quantitative_atac, sampling_step)
    
    def __getitem__(self, index):
        rrd_item = super().__getitem__(index)
        zarr_item = self.zarr_dataset[index]
        return {'rrd': rrd_item, 'zarr': zarr_item}

class InferenceEverythingDataset(InferenceReferenceRegionDataset):
    def __init__(self, reference_region_motif: ReferenceRegionMotif,
                 zarr_dataset: InferenceDataset,
                 quantitative_atac: bool = False,
                 sampling_step: int = 50) -> None:
        super().__init__(reference_region_motif, zarr_dataset, quantitative_atac, sampling_step)

    def __getitem__(self, index):
        rrd_item = super().__getitem__(index)
        zarr_item = self.zarr_dataset[index]
        return {'rrd': rrd_item, 'zarr': zarr_item}