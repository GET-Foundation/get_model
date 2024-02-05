import logging
import os
import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import zarr
from scipy.sparse import coo_matrix, load_npz, vstack
from torch.utils.data import Dataset
from tqdm import tqdm

from get_model.dataset.augmentation import (DataAugmentationForGETPeak,
                                            DataAugmentationForGETPeakFinetune)
from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.io import (generate_paths, get_hierachical_ctcf_pos,
                                  prepare_sequence_idx)
from get_model.dataset.splitter import cell_splitter, chromosome_splitter
from get_model.dataset.zarr_dataset import \
    PretrainDataset as ZarrPretrainDataset
from get_model.dataset.zarr_dataset import DenseZarrIO

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

        celltype_metadata_path = os.path.join(
            self.root, "data/cell_type_pretrain_human_bingren_shendure_apr2023.txt"
        )
        data_path = os.path.join(self.root, "")
        ctcf_path = os.path.join(
            self.root,
            "data/ctcf_motif_count.num_celltype_gt_5.feather",
        )
        ctcf = pd.read_feather(ctcf_path)
        # Chromosome	Start	End	num_celltype	strand_positive	strand_negative
        # 0	chr1	267963	268130	43	1.0	1.0
        # 1	chr1	586110	586234	9	2.0	1.0
        # 2	chr1	609306	609518	8	2.0	0.0
        # 3	chr1	610547	610776	14	3.0	0.0
        # 4	chr1	778637	778892	83	0.0	1.0

        (
            peaks,
            seqs,
            cells,
            _,
            _,
            ctcf_pos,
        ) = make_dataset(
            True,
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

        if seqs is not None:
            self.use_seq = True
        else:
            self.use_seq = False

        if len(peaks) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        if is_train:
            print("total train samples:", len(peaks))
        else:
            print("total test samples:", len(peaks))

        self.peaks = peaks
        self.seqs = seqs
        self.cells = cells
        self.ctcf_pos = ctcf_pos

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is cell_index of the target cell.
        """
        peak = self.peaks[index]
        if self.use_seq:
            seq = self.seqs[index]
        else:
            seq = None
        cell = self.cells[index]
        ctcf_pos = self.ctcf_pos[index]

        if self.transform is not None:
            peak, seq, mask = self.transform(peak, seq)

        if peak.shape[0] == 1:
            peak = peak.squeeze(0)

        # seq: [200, 1000, 4]
        # peak: [200, 1000, 1]

        seq = torch.rand(peak.shape[0], 1000, 4)      # [200, 1000, 4]
        peak = torch.rand(peak.shape[0], 1000, 1)     # [200, 1000, 1]

        return peak, seq, mask, ctcf_pos

    def __len__(self) -> int:
        return len(self.peaks)


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

        celltype_metadata_path = os.path.join(
            self.root, "data/cell_type_pretrain_human_bingren_shendure_apr2023.txt"
        )
        data_path = os.path.join(self.root, "")
        ctcf_path = os.path.join(
            self.root,
            "data/ctcf_motif_count.num_celltype_gt_5.feather",
        )
        ctcf = pd.read_feather(ctcf_path)
        # Chromosome	Start	End	num_celltype	strand_positive	strand_negative
        # 0	chr1	267963	268130	43	1.0	1.0
        # 1	chr1	586110	586234	9	2.0	1.0
        # 2	chr1	609306	609518	8	2.0	0.0
        # 3	chr1	610547	610776	14	3.0	0.0
        # 4	chr1	778637	778892	83	0.0	1.0

        (
            peaks,
            seqs,
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
        if seqs is not None:
            self.use_seq = True
        else:
            self.use_seq = False

        if len(peaks) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)

        if is_train:
            print("total train samples:", len(peaks))
        else:
            print("total test samples:", len(peaks))

        self.peaks = peaks
        self.seqs = seqs
        # self.cells = cells
        self.targets = targets
        self.ctcf_pos = ctcf_pos
        self.tssidxs = np.array(tssidx)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is cell_index of the target cell.
        """
        peak = self.peaks[index]
        if self.use_seq:
            seq = self.seqs[index]
        else:
            seq = None
        target = self.targets[index]
        tssidx = self.tssidxs[index]

        # cell = self.cells[index]
        ctcf_pos = self.ctcf_pos[index]

        if self.transform is not None:
            peak, seq, mask, target = self.transform(peak, seq, tssidx, target)
        if peak.shape[0] == 1:
            peak = peak.squeeze(0)
        return peak, seq, mask, ctcf_pos, target

    def __len__(self) -> int:
        return len(self.peaks)


def build_dataset(is_train, args):
    if args.data_set == "Pretrain":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        dataset = PretrainDataset(
            args.data_path,
            num_region_per_sample=args.num_region_per_sample,
            is_train=is_train,
            transform=transform,
            args=args,
        )

    elif args.data_set == "Expression":
        transform = DataAugmentationForGETPeakFinetune(args)
        dataset = ExpressionDataset(
            args.data_path,
            num_region_per_sample=args.num_region_per_sample,
            is_train=is_train,
            transform=transform,
            args=args,
        )

    else:
        raise NotImplementedError()

    return dataset

def setup_zarr_dataset(zarr_list, is_train, args, sequence_obj=None, dataset_size=40960):
    # get FILEPATH
    root = args.data_path
    codebase = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    if sequence_obj is None:
        sequence_obj = DenseZarrIO(f'{args.data_path}/hg38.zarr', dtype='int8')
        sequence_obj.load_to_memory_dense()
    else:
        logging.info('sequence_obj is provided')
    dataset = ZarrPretrainDataset(zarr_list,
        f'{root}/hg38.zarr',
        f'{root}/hg38_motif_result.zarr', [
        f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', 
        f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], 
        peak_name=args.peak_name, insulation_subsample_ratio=0.8,
        additional_peak_columns=['Expression_positive', 'Expression_negative'], preload_count=args.preload_count, 
        n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target,
        n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, 
        sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes, leave_out_chromosomes=args.leave_out_chromosomes, 
        is_train=is_train, non_redundant=args.non_redundant, filter_by_min_depth=args.filter_by_min_depth, dataset_size=dataset_size)
    # log all settings
    logging.info('data_path: %s' % str(args.data_path))
    logging.info('zarr_list: %s' % str(zarr_list))
    logging.info('peak_name: %s' % str(args.peak_name))
    logging.info('insulation_subsample_ratio: %s' % str(0.8))
    logging.info('additional_peak_columns: %s' % str(['Expression_positive', 'Expression_negative']))
    logging.info('preload_count: %s' % str(args.preload_count))
    logging.info('n_packs: %s' % str(args.n_packs))
    logging.info('max_peak_length: %s' % str(args.max_peak_length))
    logging.info('center_expand_target: %s' % str(args.center_expand_target))
    logging.info('n_peaks_lower_bound: %s' % str(args.n_peaks_lower_bound))
    logging.info('n_peaks_upper_bound: %s' % str(args.n_peaks_upper_bound))
    logging.info('use_insulation: %s' % str(args.use_insulation))
    logging.info('sequence_obj: %s' % str(sequence_obj))
    logging.info('leave_out_celltypes: %s' % str(args.leave_out_celltypes))
    logging.info('leave_out_chromosomes: %s' % str(args.leave_out_chromosomes))
    logging.info('is_train: %s' % str(is_train))
    logging.info('non_redundant: %s' % str(args.non_redundant))
    logging.info('filter_by_min_depth: %s' % str(args.filter_by_min_depth))
    logging.info('dataset_size: %s' % str(dataset_size))

    return dataset



def build_dataset_zarr(is_train, args, sequence_obj=None):
    if is_train and args.data_set == "Pretrain":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
        dataset = ZarrPretrainDataset([f'{root}/shendure_fetal_dense.zarr',
                            f'{root}/bingren_adult_dense.zarr'],
                           f'{root}/hg38.zarr',
                           f'{root}/hg38_motif_result.zarr', [
                           f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', 
                           f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], 
                           peak_name=args.peak_name, preload_count=args.preload_count, insulation_subsample_ratio=0.8, n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target, n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes,
                           leave_out_chromosomes=args.leave_out_chromosomes, is_train=is_train, dataset_size=40960, )
    elif not is_train and args.eval_data_set == "Pretrain.GBM_eval":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        # codebase = os.path.dirname(os.path.dirname(
        #    os.path.dirname(os.path.abspath(__file__))))
        codebase = "/pmglocal/alb2281/repos/get_model"
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
        dataset = ZarrPretrainDataset([
            f'{root}/htan_gbm_dense.zarr',
                           ],
                           f'{root}/hg38.zarr',
                           f'{root}/hg38_motif_result.zarr', [
                           f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', 
                           f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], 
                           peak_name=args.peak_name, preload_count=args.preload_count, insulation_subsample_ratio=0.8, n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target, n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes,
                           leave_out_chromosomes=args.leave_out_chromosomes,  is_train=is_train, dataset_size=4096)
    elif is_train and args.data_set == "Expression_Finetune_Fetal":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        # codebase = os.path.dirname(os.path.dirname(
        #    os.path.dirname(os.path.abspath(__file__))))
        codebase = "/pmglocal/alb2281/repos/get_model"
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
        logging.info('Using Expression_Finetune_Fetal')
        dataset = ZarrPretrainDataset([
            f'{root}/shendure_fetal_dense.zarr',
            ],
                           f'{root}/hg38.zarr',
                           f'{root}/hg38_motif_result.zarr', [
                           f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', 
                           f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], 
                           peak_name=args.peak_name, insulation_subsample_ratio=0.8,
                           additional_peak_columns=['Expression_positive', 'Expression_negative'], preload_count=args.preload_count, 
                           n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target,
                           n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, 
                           sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes, leave_out_chromosomes=args.leave_out_chromosomes, 
                           is_train=is_train, non_redundant=args.non_redundant, filter_by_min_depth=args.filter_by_min_depth, dataset_size=65536)
    elif not is_train and args.eval_data_set == "Expression_Finetune_Fetal.fetal_eval":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')

        logging.info('Using Expression_Finetune_Fetal.fetal_eval')
        dataset = ZarrPretrainDataset([
            f'{root}/shendure_fetal_dense.zarr',
            ],
                           f'{root}/hg38.zarr',
                           f'{root}/hg38_motif_result.zarr', [
                           f'{codebase}/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', 
                           f'{codebase}/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], 
                           peak_name=args.peak_name, insulation_subsample_ratio=0.8, additional_peak_columns=['Expression_positive', 'Expression_negative'], preload_count=args.preload_count, 
                           n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target, n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes,
                           leave_out_chromosomes=None, is_train=is_train, non_redundant=None, filter_by_min_depth=None, dataset_size=4096)
    elif is_train and args.data_set == "HTAN_GBM":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')
        logging.info('Using HTAN_GBM') 
        dataset = ZarrPretrainDataset([
            f'{root}/htan_gbm_dense.zarr',
            ],
                           f'/pmglocal/alb2281/get_data/get_resources/hg38.zarr',
                           f'/pmglocal/alb2281/get_data/get_resources/hg38_motif_result.zarr', 
                           ['/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], 
                           peak_name=args.peak_name, insulation_subsample_ratio=0.8,
                           additional_peak_columns=['Expression_positive', 'Expression_negative'], preload_count=args.preload_count, 
                           n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target,
                           n_peaks_lower_bound=args.n_peaks_lower_bound, n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation,
                           sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes, leave_out_chromosomes=args.leave_out_chromosomes,
                           is_train=is_train, non_redundant=args.non_redundant, filter_by_min_depth=args.filter_by_min_depth, dataset_size=65536)
    elif not is_train and args.eval_data_set == "HTAN_GBM.eval":
        transform = DataAugmentationForGETPeak(args)
        print("Data Aug = %s" % str(transform))
        root = args.data_path
        # get FILEPATH
        codebase = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if sequence_obj is None:
            sequence_obj = DenseZarrIO(f'{root}/hg38.zarr', dtype='int8')
            sequence_obj.load_to_memory_dense()
        else:
            logging.info('sequence_obj is provided')

        logging.info('Using HTAN_GBM.eval')
        dataset = ZarrPretrainDataset([
            f"{root}/htan_gbm_dense.zarr",
            ],
                           f'/pmglocal/alb2281/get_data/get_resources/hg38.zarr',
                           f'/pmglocal/alb2281/get_data/get_resources/hg38_motif_result.zarr', 
                           ['/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], 
                           peak_name=args.peak_name, insulation_subsample_ratio=0.8, additional_peak_columns=['Expression_positive', 'Expression_negative'], preload_count=args.preload_count, 
                           n_packs=args.n_packs, max_peak_length=args.max_peak_length, center_expand_target=args.center_expand_target, n_peaks_lower_bound=args.n_peaks_lower_bound, 
                           n_peaks_upper_bound=args.n_peaks_upper_bound, use_insulation=args.use_insulation, sequence_obj=sequence_obj, leave_out_celltypes=args.leave_out_celltypes,
                           leave_out_chromosomes=args.leave_out_chromosomes, is_train=is_train, non_redundant=None, filter_by_min_depth=None, dataset_size=4096)
        
    else:
        raise NotImplementedError()

    return dataset


def make_dataset(
    is_pretrain: bool,
    datatypes: str,
    data_path: str,
    celltype_metadata_path: str,
    num_region_per_sample: int,
    leave_out_celltypes: str,
    leave_out_chromosomes: str,
    use_natac: bool,
    use_seq: bool,
    is_train: bool,
    ctcf: pd.DataFrame,
    step: int = 200,
) -> Tuple[
    List[coo_matrix], List[np.ndarray], List[str], List[coo_matrix], List[np.ndarray], List[np.ndarray]
]:
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
        step (int, optional): Step size for generating samples. Defaults to 200.

    Returns:
        Tuple[List[ATACSample], List[str], List[coo_matrix], List[np.ndarray], List[np.ndarray]]: A tuple containing the generated dataset,
        cell labels, target data, TSS indices, and CTCF position segmentation.
    """
    celltype_metadata = pd.read_csv(celltype_metadata_path, sep=",", dtype=str)

    # generate file id list
    file_id_list, cell_dict, datatype_dict = cell_splitter(
        celltype_metadata,
        leave_out_celltypes,
        datatypes,
        is_train=is_train,
        is_pretrain=is_pretrain,
    )

    # initialize lists
    peak_list = []
    if use_seq:
        seq_list = []
    else:
        seq_list = None
    cell_list = []
    if not is_pretrain:
        target_list = []
        tssidx_list = []
    else:
        target_list = None
        tssidx_list = None
    ctcf_pos_list = []

    for file_id in tqdm(file_id_list):
        print(file_id)
        cell_label = cell_dict[file_id]
        data_type = datatype_dict[file_id]

        # generate file paths
        paths_dict = generate_paths(
            file_id, data_path, data_type, use_natac=use_natac)

        # read celltype peak annotation files
        celltype_annot = pd.read_csv(paths_dict["celltype_annot_csv"], sep=",")

        # prepare sequence idx
        if use_seq:
            if not os.path.exists(paths_dict["seq_npz"]):
                continue
            seq_data = zarr.load(paths_dict["seq_npz"])['arr_0']
            celltype_annot = prepare_sequence_idx(
                celltype_annot, num_region_per_sample)

        # Compute sample specific CTCF position segmentation
        # ctcf_pos = get_hierachical_ctcf_pos(
        #     celltype_annot, ctcf, cut=[5, 10, 20, 50, 100, 200]
        # )
        ctcf_pos = None

        # load data
        try:
            peak_data = load_npz(paths_dict["peak_npz"])
            print("feature shape:", peak_data.shape)
        except:
            print("File not exist - FILE ID: ", file_id)
            continue

        if not is_pretrain:
            target_data = np.load(paths_dict["target_npy"])
            tssidx_data = np.load(paths_dict["tssidx_npy"])
            print("target shape:", target_data.shape)

        # Get input chromosomes
        all_chromosomes = celltype_annot["Chromosome"].unique().tolist()
        input_chromosomes = chromosome_splitter(
            all_chromosomes, leave_out_chromosomes, is_train=is_train
        )
        print('input_chromosomes:', input_chromosomes)

        # Generate sample list
        for chromosome in input_chromosomes:
            idx_peak_list = celltype_annot.index[
                celltype_annot["Chromosome"] == chromosome
            ].tolist()
            idx_peak_start = idx_peak_list[0]
            idx_peak_end = idx_peak_list[-1]
            # NOTE: overlapping split chrom
            for i in range(idx_peak_start, idx_peak_end, step):
                # add some randomization
                shift = np.random.randint(-step//2, step//2)
                start_index = max(0, i + shift)
                end_index = start_index + num_region_per_sample

                # sanity check
                if end_index > idx_peak_end:
                    print("end_index > idx_peak_end")
                    continue
                celltype_annot_i = celltype_annot.iloc[start_index:end_index, :]
                if celltype_annot_i.iloc[-1].End - celltype_annot_i.iloc[0].Start > 5000000:
                    # change end_index to avoid too large region
                    # find the region that is < 5000000 apart from the start
                    end_index = celltype_annot_i[celltype_annot_i.End -
                                                 celltype_annot_i.Start < 5000000].index[-1]
                if celltype_annot_i["Start"].min() < 0:
                    continue

                # data generation
                peak_data_i = coo_matrix(peak_data[start_index:end_index])
                if use_seq:
                    # old loading mechanism when using sparse npz
                    # seq_start_idx = celltype_annot_i["SeqStartIdx"].min()
                    # seq_end_idx = celltype_annot_i["SeqEndIdx"].max()
                    seq_data_i = seq_data[start_index:end_index]
                # sample_data_i = ATACSample(sample_data_i, seq_data_i)

                # ctcf_pos_i = ctcf_pos[start_index:end_index]
                # ctcf_pos_i = ctcf_pos_i - ctcf_pos_i.min(0, keepdims=True)  # (200,5)
                ctcf_pos_i = 0
                if not is_pretrain:
                    target_i = coo_matrix(target_data[start_index:end_index])
                    tssidx_i = tssidx_data[start_index:end_index]

                if peak_data_i.shape[0] == num_region_per_sample:
                    peak_list.append(peak_data_i)
                    if use_seq:
                        seq_list.append(seq_data_i)
                    cell_list.append(cell_label)
                    if not is_pretrain:
                        target_list.append(target_i)
                        tssidx_list.append(tssidx_i)
                    ctcf_pos_list.append(ctcf_pos_i)

                else:
                    # TODO: add padding
                    continue
                    # padding to make sure the length is the same
                    # pad_len = num_region_per_sample - len(peak_data_i)
                    # peak_data_i = vstack(
                    #     [peak_data_i, coo_matrix((pad_len, peak_data_i.shape[1]))]
                    # )
                    # if use_seq:
                    #     seq_data_i = np.vstack(
                    #         [seq_data_i, np.zeros((pad_len, seq_data_i.shape[1]))]
                    #     )
                    # ctcf_pos_i = np.vstack(
                    #     [ctcf_pos_i, np.zeros((pad_len, ctcf_pos_i.shape[1]))])

    if seq_list is None:
        return peak_list, np.zeros((len(peak_list))), cell_list, target_list, tssidx_list, ctcf_pos_list
    return peak_list, seq_list, cell_list, target_list, tssidx_list, ctcf_pos_list
