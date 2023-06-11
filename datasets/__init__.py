import os
import os.path
from typing import Any, Callable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from datasets.augmentation import DataAugmentationForGETPeak, DataAugmentationForGETSequence
from datasets.io import generate_paths, get_hierachical_ctcf_pos, prepare_sequence_idx
from datasets.splitter import cell_splitter, chromosome_splitter
from scipy.sparse import coo_matrix, load_npz
from torch.utils.data import Dataset
from tqdm import tqdm


class PeaksSequence(object):
    """A tuple containing a concatenated sequence and segmentation idx for NUM_PEAKS peaks."""

    def __init__(self, sequence, region):
        """
        Initialize PeaksSequence.

        Args:
            sequence: The concatenated sequence.
            region: The region containing segmentation indices.
        """
        self.sequence = sequence
        self.starts = region["SeqStartIdx"]
        self.ends = region["SeqEndIdx"]
        self.region = region

    def __getitem__(self, index):
        start = self.starts[index]
        end = self.ends[index]
        return self.sequence[start:end]

    def __len__(self):
        return len(self.starts)

    def __repr__(self):
        return self.region.__repr__()


class ATACSample(object):
    """Object contains peak motif vector and peak sequence of a single sample."""
    def __init__(self, peak_motif, peak_sequence=None):
        self.peak_motif = peak_motif
        self.peak_sequence = peak_sequence

    def __repr__(self):
        return f'ATACSample(Number of peaks: {len(self.peak_motif)})'


class PretrainDataset(Dataset):

    """
    Args:
        root (string): Root directory path.
        num_region_per_sample (integer): number of regions for each sample
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.Normalize`` for regions.
     Attributes:
        samples (list): List of (sample, cell_index) tuples
        targets (list): The cell_index value for each sample in the dataset
    """

    def __init__(
        self,
        root: str,
        num_region_per_sample: int,
        is_train: bool = True,
        transform: Optional[Callable] = None,
        args: Optional[Any] = None,
    ) -> None:
        super(PretrainDataset, self).__init__()

        self.root = root
        self.transform = transform

        self.target_type = args.target_type
        self.target_thresh = args.target_thresh

        celltype_metadata_path = os.path.join(
            "./data/cell_type_pretrain_human_bingren_shendure_apr2023.txt"
        )
        data_path = os.path.join(self.root, "pretrain_human_bingren_shendure_apr2023")
        ctcf_path = os.path.join(
            "./data/ctcf_motif_count.num_celltype_gt_5.feather",
        )
        ctcf = pd.read_feather(ctcf_path)
        # Chromosome	Start	End	num_celltype	strand_positive	strand_negative
        # 0	chr1	267963	268130	43	1.0	1.0
        # 1	chr1	586110	586234	9	2.0	1.0
        # 2	chr1	609306	609518	8	2.0	0.0
        # 3	chr1	610547	610776	14	3.0	0.0
        # 4	chr1	778637	778892	83	0.0	1.0

        (
            samples,
            cells,
            targets,
            tssidx,
            ctcf_pos,
        ) = make_dataset(
            False,
            args.data_type,
            data_path,
            celltype_metadata_path,
            num_region_per_sample,
            leave_out_celltypes="",
            leave_out_chromosomes="",
            use_natac=False,
            use_seq=True,
            is_train=is_train,
            ctcf=ctcf,
            step=args.sampling_step,
        )
        self.tssidx = np.array(tssidx)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        if is_train:
            print("total train samples:", len(samples))
        else:
            print("total test samples:", len(samples))

        self.samples = samples
        self.cells = cells
        self.targets = targets
        self.ctcf_pos = ctcf_pos

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is cell_index of the target cell.
        """
        sample = self.samples[index]
        target = self.targets[index]

        cell = self.cells[index]
        ctcf_pos = self.ctcf_pos[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if sample.shape[0] == 1:
            sample = sample.squeeze(0)

        return sample, target, cell, ctcf_pos

    def __len__(self) -> int:
        return len(self.samples)


class ExpressionDataset(Dataset):

    """
    Args:
        root (string): Root directory path.
        num_region_per_sample (integer): number of regions for each sample
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.Normalize`` for regions.
     Attributes:
        samples (list): List of (sample, cell_index) tuples
        targets (list): The cell_index value for each sample in the dataset
    """

    def __init__(
        self,
        root: str,
        num_region_per_sample: int,
        is_train: bool = True,
        transform: Optional[Callable] = None,
        args: Optional[Any] = None,
    ) -> None:
        super(ExpressionDataset, self).__init__()

        self.root = root
        self.transform = transform

        self.target_type = args.target_type
        self.target_thresh = args.target_thresh

        celltype_metadata_path = os.path.join(
            "./data/cell_type_pretrain_human_bingren_shendure_apr2023.txt"
        )
        data_path = os.path.join(self.root, "pretrain_human_bingren_shendure_apr2023")
        ctcf_path = os.path.join(
            "./data/ctcf_motif_count.num_celltype_gt_5.feather",
        )
        ctcf = pd.read_feather(ctcf_path)
        # Chromosome	Start	End	num_celltype	strand_positive	strand_negative
        # 0	chr1	267963	268130	43	1.0	1.0
        # 1	chr1	586110	586234	9	2.0	1.0
        # 2	chr1	609306	609518	8	2.0	0.0
        # 3	chr1	610547	610776	14	3.0	0.0
        # 4	chr1	778637	778892	83	0.0	1.0

        (
            samples,
            cells,
            targets,
            tssidx,
            ctcf_pos,
        ) = make_dataset(
            False,
            args.data_type,
            data_path,
            celltype_metadata_path,
            num_region_per_sample,
            args.leave_out_celltypes,
            args.leave_out_chromosomes,
            args.use_natac,
            args.use_seq,
            is_train,
            ctcf,
            args.sampling_step,
        )
        self.tssidx = np.array(tssidx)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        if is_train:
            print("total train samples:", len(samples))
        else:
            print("total test samples:", len(samples))

        self.samples = samples
        self.cells = cells
        self.targets = targets
        self.ctcf_pos = ctcf_pos

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is cell_index of the target cell.
        """
        sample = self.samples[index]  # (1, 600, 111)
        target = self.targets[index]  # (600, )

        cell = self.cells[index]
        ctcf_pos = self.ctcf_pos[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if sample.shape[0] == 1:
            sample = sample.squeeze(0)

        return sample, target, cell, ctcf_pos

    def __len__(self) -> int:
        return len(self.samples)


def build_dataset(is_train, args):
    if args.data_set == "Pretrain":
        if args.use_seq:
            transform = DataAugmentationForGETSequence(args)
        else:
            transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        dataset = PretrainDataset(
            args.data_path,
            num_region_per_sample=args.num_region_per_sample,
            is_train=is_train,
            transform=None,
            args=args,
        )

    elif args.data_set == "Expression":
        dataset = ExpressionDataset(
            args.data_path,
            num_region_per_sample=args.num_region_per_sample,
            is_train=is_train,
            transform=None,
            args=args,
        )

    else:
        raise NotImplementedError()

    return dataset


def make_dataset(
    is_pretrain: bool,
    datatypes: str,
    data_path: str,
    celltype_metadata: str,
    num_region_per_sample: int,
    leave_out_celltypes: str,
    leave_out_chromosomes: str,
    use_natac: bool,
    use_seq: bool,
    is_train: bool,
    ctcf: pd.DataFrame,
    step: int = 50,
) -> Tuple[List[ATACSample], List[str], List[coo_matrix], List[np.ndarray], List[np.ndarray]]:
    """
    Generates a dataset for training or testing.

    Args:
        is_pretrain (bool): Whether it is a pretraining dataset.
        datatypes (str): String of comma-separated data types.
        data_path (str): Path to the data.
        celltype_metadata (str): Path to the celltype metadata file.
        num_region_per_sample (int): Number of regions per sample.
        leave_out_celltypes (str): String of comma-separated cell types to leave out.
        leave_out_chromosomes (str): String of comma-separated chromosomes to leave out.
        use_natac (bool): Whether to use peak data with no ATAC count values.
        use_seq (bool): Whether to use sequence data.
        is_train (bool): Whether it is a training dataset.
        ctcf (pd.DataFrame): CTCF data.
        step (int, optional): Step size for generating samples. Defaults to 50.

    Returns:
        Tuple[List[ATACSample], List[str], List[coo_matrix], List[np.ndarray], List[np.ndarray]]: A tuple containing the generated dataset, 
        cell labels, target data, TSS indices, and CTCF position segmentation.
    """
    celltype_metadata = pd.read_csv(celltype_metadata, sep=",")
    leave_out_celltypes = leave_out_celltypes.split(",")
    datatypes = datatypes.split(",")
    leave_out_chromosomes = leave_out_chromosomes.split(",")

    # generate file id list
    file_id_list, cell_dict, datatype_dict = cell_splitter(
        celltype_metadata,
        leave_out_celltypes,
        datatypes,
        is_train=is_train,
        is_pretrain=is_pretrain
        )

    for file_id in tqdm(file_id_list):
        cell_label = cell_dict[file_id]
        data_type = datatype_dict[file_id]

        # generate file paths
        paths_dict = generate_paths(
            file_id, data_path, data_type, use_natac=use_natac
        )

        # read celltype peak annotation files
        celltype_annot = pd.read_csv(paths_dict['celltype_annot_csv'], sep=",")
        
        # prepare sequence idx
        if use_seq:
            if not os.path.exists(paths_dict['seq_npz']):
                continue
            seq_data = load_npz(paths_dict['seq_npz'])
            celltype_annot = prepare_sequence_idx(celltype_annot, num_region_per_sample)

        # Compute sample specific CTCF position segmentation
        ctcf_pos = get_hierachical_ctcf_pos(
            celltype_annot, ctcf, cut=[5, 10, 20, 50, 100, 200]
        )

        # load data
        try:
            peak_data = load_npz(paths_dict['peak_npz'])
            print("feature shape:", peak_data.shape)
        except:
            print("File not exist - FILE ID: ", file_id)
            continue

        if not is_pretrain:
            target_data = np.load(paths_dict['target_npy'])
            tssidx_data = np.load(paths_dict['tssidx_npy'])
            print("target shape:", target_data.shape)

        # Get input chromosomes
        all_chromosomes = celltype_annot["Chromosome"].unique().tolist()
        input_chromosomes = chromosome_splitter(
            all_chromosomes, leave_out_chromosomes, is_train=is_train
        )

        sample_list = []
        cell_list = []
        target_list = []
        ctcf_pos_list = []
        tssidx_list = []

        # Generate sample list
        for chromosome in input_chromosomes:
            idx_sample_list = celltype_annot.index[
                celltype_annot["Chromosome"] == chromosome
            ].tolist()
            idx_sample_start = idx_sample_list[0]
            idx_sample_end = idx_sample_list[-1]
            # NOTE: overlapping split chrom
            for i in range(idx_sample_start, idx_sample_end, step):
                start_index = i
                end_index = i + num_region_per_sample

                sample_data_i = coo_matrix(peak_data[start_index:end_index])
                if use_seq:
                    celltype_annot_i = celltype_annot.iloc[start_index:end_index, :]
                    seq_start_idx = celltype_annot_i["SeqStartIdx"].min()
                    seq_end_idx = celltype_annot_i["SeqEndIdx"].max()
                    seq_data_i = PeaksSequence(seq_data[seq_start_idx:seq_end_idx, :], celltype_annot_i)
                else:
                    seq_data_i = None
                sample_data_i = ATACSample(sample_data_i, seq_data_i)
                target_i = coo_matrix(target_data[start_index:end_index])
                ctcf_pos_i = ctcf_pos[start_index:end_index]
                ctcf_pos_i = ctcf_pos_i - ctcf_pos_i.min(0, keepdims=True)  # (200,5)
                tssidx_i = tssidx_data[start_index:end_index]

                if sample_data_i.shape[0] == num_region_per_sample:
                    sample_list.append(sample_data_i)
                    cell_list.append(cell_label)
                    target_list.append(target_i)
                    ctcf_pos_list.append(ctcf_pos_i)
                    tssidx_list.append(tssidx_i)
                else:
                    continue

    return sample_list, cell_list, target_list, tssidx_list, ctcf_pos_list
