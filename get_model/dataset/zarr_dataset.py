import logging
import os.path
import sys
import warnings
from posixpath import basename
from caesar.io.zarr_io import DenseZarrIO
import numpy as np
import pandas as pd
import torch
import zarr
from caesar.io.zarr_io import CelltypeDenseZarrIO, DenseZarrIO
from pyranges import PyRanges as pr
from scipy.sparse import csr_matrix, vstack
from torch.utils.data import Dataset
from tqdm import tqdm


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Suppress all deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append('/pmglocal/xf2217/get_model')

def get_padding_pos(mask):
    mask_ = mask.clone()
    mask_[mask_!=-10000]=0
    mask_[mask_!=0]=1
    return mask_

def get_mask_pos(mask):
    mask_ = mask.clone()
    mask_[mask_==-10000]=0
    return mask_

def get_sequence_with_mutations(arr, start, end, mut):
    from atac_rna_data_processing.io.sequence import DNASequence
    arr_wt = arr.copy()
    mut_df_chr = mut.query('Start>=@start & End <= @end').sort_values('Start')
    
    offset = 0  # Track the net offset introduced by mutations

    for _, row in tqdm(mut_df_chr.iterrows()):
        # Adjust mutation positions by the current offset
        mut_start = row.Start - start + offset
        mut_end = row.End - start + offset
        
        # Convert reference and alternative alleles to one-hot encoded format
        wt_on_genome = arr[mut_start:mut_end]
        wt_in_mut_file = DNASequence(row.Ref).one_hot.astype(float)
        alt_sequence = DNASequence(row.Alt).one_hot.astype(float)
        
        # Ensure the sequence in the genome matches the reference sequence
        if not np.array_equal(wt_on_genome, wt_in_mut_file):
            logging.warning(f"Reference sequence in mutation file does not match the genome sequence at {row.Start}-{row.End}. Skipping this mutation.")
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
                padded_track = csr_matrix((end-start+2*padding, track_depth), dtype=track_dtype)
            stacked_track_list.append(padded_track)
        else:
            # Create a zero array of the required shape for inactivated peaks.
            zero_array = csr_matrix((end-start+2*padding, track_depth), dtype=track_dtype)
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
    hic_idx = np.array([row.Start // resolution - start + 1 for _, row in csv_region.iterrows()])
    mzd = hic.getMatrixZoomData(chrom, chrom, method, "KR", "BP", resolution)
    numpy_matrix = mzd.getRecordsAsMatrix(start * resolution, end * resolution, start * resolution, end * resolution)
    dst = np.log10(numpy_matrix[hic_idx,:][:, hic_idx]+1)
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


class ZarrDataPool(object):
    """A class to handle data loading for a slot.

    This class loads data from a set of zarr files and generates samples for training.
    Each sample consists of a track and a peak sequence extracted from a 4Mbp window.
    The track is a 2D sparse array of shape (nucleotide,1) and the peak sequence is a 2D array of shape (nucleotide,4). Peak sequence is generated by multiplying the peak mask with the genomic sequence. Track is cell-type specific ATAC-seq insertion counts per nucleotide, without any normalization. Peak sequence is a one-hot encoding of DNA. Other metadata about the sample is also included in the sample as a dictionary.

    Parameters:
    zarr_dirs (list): A list of paths to zarr files.
    genome_seq_zarr (str): Path to the genome sequence zarr file.
    insulation_paths (list): A list of paths to insulation data.
    peak_name (str): The name of the peak track in the zarr files.
    insulation_subsample_ratio (float): The ratio of insulation data to use.
    max_peak_length (int): The maximum length of peaks to include in the dataset.
    center_expand_target (int): The target length of peaks to center and expand.
    sequence_obj (DenseZarrIO): A DenseZarrIO object for genome sequence.
    motif_mean_std_obj (MotifMeanStd): A MotifMeanStd object for motif mean and std.
    additional_peak_columns (list): A list of additional peak columns to include in the dataset.
    leave_out_celltypes (list): A list of cell type IDs to leave out.
    leave_out_chromosomes (list): A list of chromosome names to leave out.
    is_train (bool): Whether to use the dataset for training.
    non_redundant (bool): Whether to remove redundant cell type instances.
    filter_by_min_depth (bool): Whether to filter out samples by minimum depth.

    Returns:
    ZarrDataPool: A ZarrDataPool object.

    Note:
    This class is used by PretrainDataset to load data from zarr files.
    """

    def __init__(self, zarr_dirs, genome_seq_zarr, insulation_paths, peak_name='peaks', insulation_subsample_ratio=0.1,
                 max_peak_length=None, center_expand_target=None, sequence_obj=None, motif_mean_std_obj=None,
                 additional_peak_columns=None, leave_out_celltypes=None, leave_out_chromosomes=None, non_redundant='max_depth', invert_peak=False, random_shift_peak=True,
                 filter_by_min_depth=None, is_train=True, hic_path=None):
        # logging.info('Initializing ZarrDataPool')
        if sequence_obj is None:
            self.sequence = DenseZarrIO(
                genome_seq_zarr, dtype='int8', mode='r')
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
        self.invert_peak = invert_peak
        self.random_shift_peak = random_shift_peak
        self.non_redundant = non_redundant
        self.filter_by_min_depth = filter_by_min_depth
        self.initialize_datasets()
        self.calculate_metadata()
        # logging.info('ZarrDataPool initialized')

    def initialize_datasets(self):
        """
        Initialize the zarr datasets and load peaks and insulation data.
        """
        self._subset_datasets()
        self.data_keys = list(self.zarr_dict.keys())
        self.peaks_dict = self._load_peaks()
        self.insulation = self._load_insulation()
        self.hic_obj = self._load_hic()

    def _load_hic(self):
        try:
            import hicstraw
            hic_obj = hicstraw.HiCFile(self.hic_path)
        except:
            logging.warning('hicstraw is not installed, cannot load hic data, or the hic file is not found')
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
                    {data_key: cdz.filter_by_min_depth(self.filter_by_min_depth)}
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
        celltype_peaks = self._query_peaks(celltype_id, chr_name, start, end, self.invert_peak, self.random_shift_peak)
        item_insulation = self._query_insulation(chr_name, start, end)

        track = self.zarr_dict[data_key].get_track(
            celltype_id, chr_name, start, end, sparse=True).T.astype(np.uint16)
        item_insulation = item_insulation.reset_index(drop=True).reset_index()
        
        celltype_peaks = celltype_peaks.reset_index(drop=True).reset_index()
        if self.motif_mean_std_obj is not None:
            motif_mean_std = self.motif_mean_std_obj.data_dict[chr_name][chr_chunk_idx:chr_chunk_idx+2].reshape(
                2, 2, -1).mean(0)

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
        chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std = self.load_data(data_key, celltype_id, chr_name, start, end)
        item_insulation['key'] = str(
            window_index) + '_' + item_insulation['index'].astype(str)
        return window_index, chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std

    def _inactivated_peaks(self, celltype_peaks, peak_inactivation):
        """
        Generate a column label for 
        inactivated peaks that are in peak_inactivation use pyranges
        """
        # double reset_index to get numeric index
        celltype_peaks_ = pr(celltype_peaks.reset_index(drop=True).reset_index())
        peak_inactivation_ = pr(peak_inactivation)
        celltype_peaks_ = celltype_peaks_.join(peak_inactivation_, how='left', suffix='_peak').df
        # get numeric index of the peaks that are in peak_inactivation
        inactivated_peak_idx = celltype_peaks_.loc[celltype_peaks_['Start_peak']!=-1]['index'].drop_duplicates().values
        return inactivated_peak_idx
    
    def _get_peak_names(self, data_key, celltype_id):
        """
        Return a list of peak names for a celltype, use glob peaks*
        """
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
                if self.center_expand_target!=0:
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

        insulation = pd.concat(insulation_list).drop_duplicates(subset=['Chromosome', 'Start', 'End'])
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

    def _query_peaks(self, celltype_id, chr_name, start, end, invert=None, random_shift=False):
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
        if random_shift:
            random_int = np.random.randint(-10, 10, size=df.shape[0])
        else:
            random_int = 0
        df['Start'] = df['Start'].values + random_int
        df['End'] = df['End'].values + random_int
        df = df.query(
            'Chromosome == @chr_name and Start >= @start and End <= @end')
        if df.shape[0]>30 and invert is not None and isinstance(invert, float) and np.random.rand() < invert:
            n_peaks = df.shape[0]
            # invert the peaks with a probability of `inverted`
            boundary = pd.DataFrame({'Chromosome': [chr_name], 'Start': [start], 'End': [end]})
            df = pr(boundary).subtract(pr(df)).tile(self.center_expand_target).sample(n_peaks).df
            df = df.query(
            'Chromosome == @chr_name and Start >= @start and End <= @end')
            for col in self.additional_peak_columns:
                df[col] = 0
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
        chunk_idx = sum([self.chrom_n_chunks[list(self.chrom_n_chunks.keys())[i]]-1 for i in range(current_chr_idx)]) + chr_chunk_idx
        return chunk_idx
    
    def generate_sample(self, chr_name, start, end, data_key, celltype_id, track, celltype_peaks, motif_mean_std,
                         mut=None, peak_inactivation=None, padding=50):
        """
        Generate a single sample.
        """
        chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std = self.load_data(
            data_key, celltype_id, chr_name, start, end)
        track_start = celltype_peaks['Start'].min() - padding
        track_end = celltype_peaks['End'].max() + padding
        # peak_inactivation is a dataframe of peaks to inactivate
        # overlap with celltype_peaks to keep the peaks that are not in peak_inactivation
        if peak_inactivation is not None:
            inactivated_peak_idx = self._inactivated_peaks(celltype_peaks, peak_inactivation)
        else:
            inactivated_peak_idx = None

                        
        if self.hic_obj is not None:
            hic_matrix = get_hic_from_idx(self.hic_obj, celltype_peaks)


        if self.additional_peak_columns is not None:
            # assume numeric columns
            additional_peak_columns_data = celltype_peaks[self.additional_peak_columns].to_numpy(
            ).astype(np.float32)
        else:
            additional_peak_columns_data = None

        sequence = self.sequence.get_track(
            chr_name, track_start, track_end, sparse=False)
    
        if mut is not None:
            # filter the mutation data with celltype_peaks
            mut_peak = pr(mut.query('Chromosome==@chr_name')).join(pr(celltype_peaks)).df
            if mut_peak.shape[0] > 0:
                sequence_mut, sequence = get_sequence_with_mutations(
                    sequence, track_start, track_end, mut_peak)
                logging.info(f"Mutated sequence for {chr_name}:{start}-{end} has been generated")
                logging.info(f"Mutated sequence is different from the original sequence: {not np.array_equal(sequence, sequence_mut)}")
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

        sample_peak_sequence = _stack_tracks_with_padding_and_inactivation(celltype_peaks, sample_peak_sequence, padding, inactivated_peak_idx)
        sample_track = _stack_tracks_with_padding_and_inactivation(celltype_peaks, sample_track, padding, inactivated_peak_idx)
        # remove atac and expression from inactivated peak
        if inactivated_peak_idx is not None:
            additional_peak_columns_data[inactivated_peak_idx, 0:3] = 0 # keep the TSS column but set aTPM and expression to 0
        
        sample_metadata = {
            'celltype_id': celltype_id, 'chr_name': chr_name,
            'start': start, 'end': end, 'i_start': _start, 'i_end': _end, 'mask_ratio': 0.5
        }

        return sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, motif_mean_std, additional_peak_columns_data, hic_matrix
                             

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
            n_peaks_lower_bound=5, n_peaks_upper_bound=200, n_peaks_sample_gap=50, use_insulation=True, window_index=None, peak_inactivation=None, mut=None):
        # logging.info('Initializing PreloadDataPack')
        self.preload_count = preload_count
        self.zarr_data_pool = zarr_data_pool
        self.preloaded_data = []
        self.insulation_peak_counts = pd.DataFrame()
        self.preloaded_data_window_indices_mapping = {}
        self.next_sample = 0
        self.padding = padding
        self.peak_inactivation = peak_inactivation
        self.mut = mut
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
            return PreloadDataPack(self.preload_count, self.zarr_data_pool, self.padding, self.mask_ratio, self.n_peaks_lower_bound, self.n_peaks_upper_bound, self.n_peaks_sample_gap, self.use_insulation, window_index, self.peak_inactivation, self.mut)

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
        window_index, chr_name, start, end, celltype_id, track, insulations, celltype_peaks, motif_mean_std = window

        if peak_start is None or peak_end is None:
            peak_start = peak_index * self.n_peaks_sample_gap
            peak_end = peak_start + self.n_peaks_upper_bound
        celltype_peaks = celltype_peaks.iloc[peak_start:peak_end]
        track_start = celltype_peaks['Start'].min() - self.padding
        track_end = celltype_peaks['End'].max() + self.padding
        return self._generate_sample(chr_name, start, end, celltype_id, track, celltype_peaks, motif_mean_std,
                                     track_start, track_end, mut)

    def _inactivated_peaks(self, celltype_peaks, peak_inactivation):
        """
        Generate a column label for 
        inactivated peaks that are in peak_inactivation use pyranges
        """
        # double reset_index to get numeric index
        celltype_peaks_ = pr(celltype_peaks.reset_index(drop=True).reset_index())
        peak_inactivation_ = pr(peak_inactivation)
        celltype_peaks_ = celltype_peaks_.join(peak_inactivation_, how='left', suffix='_peak')
        # get numeric index of the peaks that are in peak_inactivation
        inactivated_peak_idx = celltype_peaks_.loc[celltype_peaks_['Start_peak']!=-1]['index'].drop_duplicates()
        return inactivated_peak_idx
    

    
    def _generate_sample(self, chr_name, start, end, celltype_id, track, celltype_peaks, motif_mean_std,
                         track_start, track_end, mut=None):
        """
        Generate a single sample from a window.
        """
        # peak_inactivation is a dataframe of peaks to inactivate
        # overlap with celltype_peaks to keep the peaks that are not in peak_inactivation, unless the peak is a TSS
        if self.peak_inactivation is not None and self.peak_inactivation != 'random_tss':
            inactivated_peak_idx = self._inactivated_peaks(celltype_peaks, self.peak_inactivation)
        elif self.peak_inactivation == 'random_tss':
            inactivated_peak_idx = celltype_peaks.reset_index(drop=True).reset_index().query('TSS==1').sample(frac=0.1).index.values
        else:
            inactivated_peak_idx = None
        if self.additional_peak_columns is not None:
            # assume numeric columns
            additional_peak_columns_data = celltype_peaks[self.additional_peak_columns].to_numpy(
            ).astype(np.float32)
        else:
            additional_peak_columns_data = None

        sequence = self.zarr_data_pool.sequence.get_track(
            chr_name, track_start, track_end, sparse=False)
    
        if mut is not None:
            # filter the mutation data with celltype_peaks
            mut_peak = pr(mut.query('Chromosome==@chr_name')).join(pr(celltype_peaks)).df
            if mut_peak.shape[0] > 0:
                sequence_mut, sequence = get_sequence_with_mutations(
                    sequence, track_start, track_end, mut_peak)
                sequence = sequence_mut

                
        if self.zarr_data_pool.hic_obj is not None:
            hic_matrix = get_hic_from_idx(self.zarr_data_pool.hic_obj, celltype_peaks)

        celltype_peaks = celltype_peaks[[
            'Start', 'End']].to_numpy().astype(np.int64)


        sample_peak_sequence = self.zarr_data_pool._generate_peak_sequence(
            celltype_peaks, sequence, track_start, track_end)
        

        # where the track locates in the window
        _start, _end = track_start - start, track_end - start
        # where the peaks locate in the track
        celltype_peaks = celltype_peaks - track_start

        sample_track = track[_start:_end]

        sample_peak_sequence = _stack_tracks_with_padding_and_inactivation(celltype_peaks, sample_peak_sequence, self.padding, inactivated_peak_idx)
        sample_track = _stack_tracks_with_padding_and_inactivation(celltype_peaks, sample_track, self.padding, inactivated_peak_idx)
        # remove atac and expression from inactivated peak
        if inactivated_peak_idx is not None:
            additional_peak_columns_data[inactivated_peak_idx, 0:3] = 0 # keep the TSS column but set aTPM and expression to 0
        
        sample_metadata = {
            'celltype_id': celltype_id, 'chr_name': chr_name,
            'start': start, 'end': end, 'i_start': _start, 'i_end': _end, 'mask_ratio': self.mask_ratio
        }

        return sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, motif_mean_std, additional_peak_columns_data, hic_matrix

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
        window_index, chr_name, start, end, celltype_id, track, insulations, celltype_peaks, motif_mean_std = window
        if len(insulations) == 0:
            raise ValueError('Empty insulation')
        track_start, track_end = self._insulation_sampler(
            insulations, insulation_index)
        celltype_peaks = celltype_peaks.query('Start>@track_start and End<@track_end')
        if celltype_peaks.shape[0] < self.n_peaks_lower_bound:
            logging.info('No enough peaks')
        return self._generate_sample(chr_name, start, end, celltype_id, track, celltype_peaks, motif_mean_std,
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
        insulation_start, insulation_end = insulation_df.iloc[insulation_index][['Start', 'End']]
        return insulation_start, insulation_end

    def _calculate_peak_num_per_sample(self):
        """
        Calculate the number of peaks per sample.

        Returns:
        pandas.DataFrame: A pandas dataframe containing the number of peaks per sample.
        """
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

        for window_index, chr_name, start, end, celltype_id, track, item_insulation, celltype_peaks, motif_mean_std in self.preloaded_data:
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
    def __init__(self, zarr_dirs, genome_seq_zarr, genome_motif_zarr, insulation_paths, peak_name='peaks', additional_peak_columns=None, preload_count=50, padding=50, mask_ratio=0.5, n_packs=2, max_peak_length=None, center_expand_target=None, insulation_subsample_ratio=0.1, n_peaks_lower_bound=5, n_peaks_upper_bound=200, use_insulation=True, sequence_obj=None, leave_out_celltypes=None, leave_out_chromosomes=None, n_peaks_sample_gap=50, is_train=True, non_redundant=False, filter_by_min_depth=False, dataset_size=655_360, peak_inactivation=None, mut=None, invert_peak=None, random_shift_peak=True, hic_path=None):
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
        genome_motif_zarr (str): Path to the genome motif zarr file.
        insulation_paths (list): A list of paths to insulation data.
        peak_name (str): The name of the peak track in the zarr files.
        preload_count (int): The number of windows to preload.
        padding (int): The number of nucleotides to pad around each peak.
        mask_ratio (float): The ratio of nucleotides to mask in the peak sequence.
        n_packs (int): The number of data packs to preload.
        max_peak_length (int): The maximum length of peaks to include in the dataset.
        center_expand_target (int): The target length of peaks to center and expand.
        insulation_subsample_ratio (float): The ratio of insulation data to use.
        n_peaks_lower_bound (int): The lower bound of number of peaks in a sample.
        n_peaks_upper_bound (int): The upper bound of number of peaks in a sample.
        sequence_obj (DenseZarrIO): A DenseZarrIO object for genome sequence.
        leave_out_celltypes (list): A list of cell type IDs to leave out.
        leave_out_chromosomes (list): A list of chromosome names to leave out.
        is_train (bool): Whether to use the dataset for training.
        non_redundant (bool): Whether to remove redundant cell type instances.
        filter_by_min_depth (bool): Whether to filter out samples by minimum depth.
        dataset_size (int): The size of the dataset.

        Returns:
        PretrainDataset: A PretrainDataset object.
        """
        logging.info('Initializing PretrainDataset')
        # # log all parameters
        for key, value in locals().items():
            logging.info(f'{key}: {value}')
        self.preload_count = preload_count
        self.padding = padding
        self.mask_ratio = mask_ratio
        self.peak_name = peak_name
        self.insulation_subsample_ratio = insulation_subsample_ratio
        self.n_peaks_lower_bound = n_peaks_lower_bound
        self.n_peaks_upper_bound = n_peaks_upper_bound
        self.n_peaks_sample_gap = n_peaks_sample_gap
        self.invert_peak = invert_peak
        self.random_shift_peak = random_shift_peak
        self.use_insulation = use_insulation
        self.leave_out_celltypes = leave_out_celltypes
        self.leave_out_chromosomes = leave_out_chromosomes
        self.is_train = is_train
        self.non_redundant = non_redundant
        self.filter_by_min_depth = filter_by_min_depth
        self.dataset_size = dataset_size
        self.n_packs = n_packs
        self.additional_peak_columns = additional_peak_columns
        self.peak_inactivation = peak_inactivation
        self.mut = mut
        self.hic_path = hic_path
        if sequence_obj is None:
            self.sequence = DenseZarrIO(
                genome_seq_zarr, dtype='int8', mode='r')
            self.sequence.load_to_memory_dense()
        else:
            self.sequence = sequence_obj
        self.mms = MotifMeanStd(genome_motif_zarr)
        self.datapool = ZarrDataPool(zarr_dirs, genome_seq_zarr, insulation_paths, peak_name=peak_name,
                                     insulation_subsample_ratio=self.insulation_subsample_ratio, max_peak_length=max_peak_length, center_expand_target=center_expand_target, sequence_obj=self.sequence,
                                     motif_mean_std_obj=self.mms,
                                     additional_peak_columns=self.additional_peak_columns,
                                     leave_out_celltypes=self.leave_out_celltypes,
                                     leave_out_chromosomes=self.leave_out_chromosomes,
                                     is_train=self.is_train, non_redundant=self.non_redundant, filter_by_min_depth=self.filter_by_min_depth,
                                     invert_peak=self.invert_peak, random_shift_peak=self.random_shift_peak, hic_path=self.hic_path)

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

    def reload_data(self, slot, window_index=None):
        # This method runs in a separate thread
        # logging.info(f'Async reloading data for slot {index}')
        # reload by reinitializing the preload data pack and put it back to the preload_data_packs
        self.preload_data_packs[slot] = PreloadDataPack(
            preload_count=self.preload_count, zarr_data_pool=self.datapool, padding=self.padding, mask_ratio=self.mask_ratio, n_peaks_lower_bound=self.n_peaks_lower_bound, n_peaks_upper_bound=self.n_peaks_upper_bound, n_peaks_sample_gap=self.n_peaks_sample_gap, use_insulation=self.use_insulation, window_index=window_index, peak_inactivation=self.peak_inactivation, mut=self.mut)
        # self.preload_data_packs[slot].preload_data()
        # add the index back to avaliable packs
        self.avaliable_packs.append(slot)


def worker_init_fn_get(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(torch.initial_seed() + worker_id)

    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.preload_data_packs is None:
        dataset.preload_data_packs = [PreloadDataPack(
                     preload_count=dataset.preload_count, zarr_data_pool=dataset.datapool, padding=dataset.padding, mask_ratio=dataset.mask_ratio, n_peaks_lower_bound=dataset.n_peaks_lower_bound, n_peaks_upper_bound=dataset.n_peaks_upper_bound, n_peaks_sample_gap=dataset.n_peaks_sample_gap, use_insulation=dataset.use_insulation, peak_inactivation=dataset.peak_inactivation, mut=dataset.mut) for _ in range(dataset.n_packs)]

class InferenceDataset(PretrainDataset):
    def __init__(self, zarr_dirs, genome_seq_zarr, genome_motif_zarr, insulation_paths, gencode_obj, peak_name='peaks', 
                 n_peaks_upper_bound=100, sequence_obj=None, additional_peak_columns=None, center_expand_target=1000, max_peak_length=5000, use_insulation=False, random_shift_peak=False):
        super().__init__(zarr_dirs, genome_seq_zarr, genome_motif_zarr, insulation_paths, peak_name=peak_name, n_peaks_upper_bound=n_peaks_upper_bound, sequence_obj=sequence_obj, additional_peak_columns=additional_peak_columns, n_packs=1, max_peak_length=max_peak_length,  use_insulation=use_insulation, center_expand_target=center_expand_target, preload_count=1, insulation_subsample_ratio=1, random_shift_peak=random_shift_peak)
        self.gencode_obj = gencode_obj
        self.tss_chunk_idx = self._generate_tss_chunk_idx()

    def _generate_tss_chunk_idx(self):
        """Determine the windows to extract for each gene"""
        if os.path.exists(self.gencode_obj.feather_file.replace('.feather', '_tss_chunk_idx.feather')):
            self.tss_chunk_idx = pd.read_feather(self.gencode_obj.feather_file.replace('.feather', '_tss_chunk_idx.feather'))
            return self.tss_chunk_idx
        self.tss_chunk_idx = self.gencode_obj.gtf.copy()
        for i, row in tqdm(self.gencode_obj.gtf.iterrows()):
            # get window_index for each gene
            gene_chr = row.Chromosome
            gene_start = row.Start
            self.tss_chunk_idx.loc[i, 'chunk_idx'] = self.datapool._get_chunk_idx(gene_chr, gene_start)
        # save the tss_chunk_idx as feather in the same directory as the gencode file
        self.tss_chunk_idx.to_feather(self.gencode_obj.feather_file.replace('.feather', '_tss_chunk_idx.feather'))
        return self.tss_chunk_idx
    
    def _get_window_idx_for_tss_and_celltype(self, data_key, celltype_id, tss_idx):
        """Get window index for a gene and celltype"""
        chunk_idx = self.tss_chunk_idx.loc[tss_idx, 'chunk_idx']
        gene_name = self.tss_chunk_idx.loc[tss_idx, 'gene_name']
        start = self.tss_chunk_idx.loc[tss_idx, 'Start']
        return {'gene_name': gene_name,
                'window_idx': np.array([self.datapool._get_window_index_from_chunk_idx(data_key, celltype_id, chunk_idx)]), 
                'Start': start}

    def _get_window_idx_for_gene_and_celltype(self, data_key, celltype_id, gene_name):
        """Get window index for a gene and celltype"""
        gene_df = self.tss_chunk_idx.query('gene_name==@gene_name')
        chunk_idxs = gene_df['chunk_idx']
        start = gene_df['Start'].values
        return {'gene_name': gene_name,
                'window_idx': np.unique([self.datapool._get_window_index_from_chunk_idx(data_key, celltype_id, chunk_idx) for chunk_idx in chunk_idxs]), 
                'Start': start}
    
    def _get_gene_info_from_window_idx(self, window_idx):
        chunk_idx = window_idx % self.datapool.genome_chunk_length
        gene_df = self.tss_chunk_idx.query('chunk_idx==@chunk_idx or chunk_idx==@chunk_idx+1')
        return gene_df
    

    def __len__(self):
        return self.tss_chunk_idx['chunk_idx'].unique().shape[0] * self.datapool.n_celltypes

    def __getitem__(self, idx):
        celltype_idx = idx // self.tss_chunk_idx['chunk_idx'].unique().shape[0]
        chunk_idx = self.tss_chunk_idx['chunk_idx'].unique()[idx % self.tss_chunk_idx['chunk_idx'].unique().shape[0]]
        data_key, celltype_id = self.datapool._get_celltype(celltype_idx)
        window_idx = self.datapool._get_window_index_from_chunk_idx(data_key, celltype_id, chunk_idx)
        return window_idx

