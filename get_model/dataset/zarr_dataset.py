import logging
import os
import os.path
import warnings
from dataclasses import dataclass
from posixpath import basename
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from gcell.rna.gencode import Gencode
from scipy.sparse import coo_matrix, csr_matrix, load_npz
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

EXPRESSION_ATAC_CUTOFF = (
    0.05  # TODO turn this into a config parameter. also in collate for v3 GET
)

# Suppress all deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _chromosome_splitter(
    all_chromosomes: list, leave_out_chromosomes: str | list | None, is_train=True
):
    input_chromosomes = all_chromosomes.copy()
    if leave_out_chromosomes is None:
        leave_out_chromosomes = []
    elif isinstance(leave_out_chromosomes, str):
        if "," in leave_out_chromosomes:
            leave_out_chromosomes = leave_out_chromosomes.split(",")
        else:
            leave_out_chromosomes = [leave_out_chromosomes]

    if is_train or leave_out_chromosomes == [""] or leave_out_chromosomes == []:
        input_chromosomes = [
            chrom for chrom in input_chromosomes if chrom not in leave_out_chromosomes
        ]
    else:
        input_chromosomes = (
            all_chromosomes if leave_out_chromosomes == [] else leave_out_chromosomes
        )

    if isinstance(input_chromosomes, str):
        input_chromosomes = [input_chromosomes]
    print("Leave out chromosomes:", leave_out_chromosomes)
    print("Input chromosomes:", input_chromosomes)
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


def get_hic_from_idx(hic, csv, start=None, end=None, resolution=5000, method="oe"):
    # if from hic straw
    if hasattr(hic, "getMatrixZoomData"):
        if start is not None and end is not None:
            csv_region = csv.iloc[start:end]
        else:
            csv_region = csv
        chrom = csv_region.iloc[0].Chromosome.replace("chr", "")
        if chrom != csv_region.iloc[-1].Chromosome.replace("chr", ""):
            return None
        start = csv_region.iloc[0].Start // resolution
        end = csv_region.iloc[-1].End // resolution + 1
        if (end - start) * resolution > 4000000:
            return None
        hic_idx = np.array(
            [row.Start // resolution - start + 1 for _, row in csv_region.iterrows()]
        )
        mzd = hic.getMatrixZoomData(
            "chr" + chrom, "chr" + chrom, method, "SCALE", "BP", resolution
        )
        numpy_matrix = mzd.getRecordsAsMatrix(
            start * resolution, end * resolution, start * resolution, end * resolution
        )
        numpy_matrix = np.nan_to_num(numpy_matrix)
        dst = np.log10(numpy_matrix[hic_idx, :][:, hic_idx] + 1)
        return dst
    # if from cooler
    elif hasattr(hic, "matrix"):
        if start is not None and end is not None:
            csv_region = csv.iloc[start:end]
        else:
            csv_region = csv
        chrom = csv_region.iloc[0].Chromosome.replace("chr", "")
        if chrom != csv_region.iloc[-1].Chromosome.replace("chr", ""):
            return None
        start = csv_region.iloc[0].Start // resolution
        end = csv_region.iloc[-1].End // resolution + 1
        if (end - start) * resolution > 4000000:
            return None
        hic_idx = np.array(
            [row.Start // resolution - start for _, row in csv_region.iterrows()]
        )
        numpy_matrix = hic.matrix(balance=True).fetch(
            f"chr{chrom}:{start * resolution}-{end * resolution}"
        )
        numpy_matrix = np.nan_to_num(numpy_matrix)
        dst = np.log10(numpy_matrix[hic_idx, :][:, hic_idx] + 1)
        return dst


def get_gencode_obj(genome_seq_zarr: dict | str):
    """
    Get Gencode object for genome sequence.
    """
    # TODO: make this more flexible
    version_mapping = {
        "hg38": 44,
        "mm10": "M36",
    }
    if isinstance(genome_seq_zarr, dict):
        gencode_obj = {}
        for assembly, _ in genome_seq_zarr.items():
            gencode_obj[assembly] = Gencode(assembly, version=version_mapping[assembly])
    elif isinstance(genome_seq_zarr, str):
        assembly = basename(genome_seq_zarr).split(".")[0]
        gencode_obj = {assembly: Gencode(assembly, version=version_mapping[assembly])}
    return gencode_obj


class RegionDataset(Dataset):
    """
        PyTorch dataset class for cell type expression data.
        This is for backward compatibility with old (version 1) datasets.
    Consider use InferenceRegionMotifDataset for new datasets.

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
        metadata_path = os.path.join(self.root, metadata_path)
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
                    np.zeros(
                        int(
                            self.num_region_per_sample
                            - self.num_region_per_sample * self.mask_ratio
                        )
                    ),
                    np.ones(int(self.num_region_per_sample * self.mask_ratio)),
                ]
            )
            np.random.shuffle(mask)
        else:
            mask = tssidx
        if self.transform is not None:
            peak, mask, target = self.transform(peak, tssidx, target)
        if peak.shape[0] == 1:
            peak = peak.squeeze(0)
        return {
            "region_motif": peak.toarray().astype(np.float32),
            "mask": mask,
            "exp_label": target.toarray().astype(np.float32),
        }

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
            # logging.debug(f"Train cell types list: {celltype_list}")
            # logging.debug(f"Train data types list: {datatypes}")

            celltype_list = (
                leave_out_celltypes if leave_out_celltypes != [""] else celltype_list
            )
            logging.debug(
                f"Using validation cell type for training!!! cell types list: {celltype_list}"
            )
            logging.debug(
                f"Using validation cell type for training!!! data types list: {datatypes}"
            )

        else:
            celltype_list = (
                leave_out_celltypes if leave_out_celltypes != [""] else celltype_list
            )
            logging.debug(f"Validation cell types list: {celltype_list}")
            logging.debug(f"Validation data types list: {datatypes}")

        file_id_list = []
        datatype_dict = {}
        cell_dict = {}
        for cell in celltype_list:
            celltype_metadata_of_cell = celltype_metadata[
                celltype_metadata["celltype"] == cell
            ]
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

        logging.debug(f"File ID list: {file_id_list}")
        return file_id_list, cell_dict, datatype_dict

    @staticmethod
    def _generate_paths(
        file_id: int, data_path: str, data_type: str, quantitative_atac: bool = False
    ) -> dict:
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
        peak_npz_path = os.path.join(data_path, data_type, f"{file_id}.watac.npz")

        if not os.path.exists(peak_npz_path):
            raise FileNotFoundError(f"Peak file not found: {peak_npz_path}")

        target_npy_path = os.path.join(data_path, data_type, f"{file_id}.exp.npy")
        tssidx_npy_path = os.path.join(data_path, data_type, f"{file_id}.tss.npy")
        celltype_annot = os.path.join(data_path, data_type, f"{file_id}.csv")
        # if celltype_annot is not exist, check csv.gz
        if not os.path.exists(celltype_annot):
            celltype_annot = os.path.join(data_path, data_type, f"{file_id}.csv.gz")
        exp_feather = os.path.join(data_path, data_type, f"{file_id}.exp.feather")
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
        celltype_metadata = pd.read_csv(celltype_metadata_path, sep=",", dtype=str)
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
            logging.debug(file_id, data_path, data_type)
            paths_dict = self._generate_paths(
                file_id, data_path, data_type, quantitative_atac=quantitative_atac
            )

            celltype_peak_annot = pd.read_csv(
                paths_dict["celltype_annot_csv"], sep=","
            )  # .drop('index', axis=1)

            try:
                peak_data = load_npz(paths_dict["peak_npz"])
                logging.debug(f"Feature shape: {peak_data.shape}")
            except FileNotFoundError:
                logging.debug(f"File not found - FILE ID: {file_id}")
                continue

            if not is_pretrain:
                target_data = np.load(paths_dict["target_npy"])
                tssidx_data = np.load(paths_dict["tssidx_npy"])
                logging.debug(f"Target shape: {target_data.shape}")
                atac_cutoff = (
                    1 - (peak_data[:, 282] > EXPRESSION_ATAC_CUTOFF).toarray().flatten()
                )
                target_data[atac_cutoff, :] = 0

            if quantitative_atac is False:
                peak_data[:, 282] = 1

            all_chromosomes = celltype_peak_annot["Chromosome"].unique().tolist()
            input_chromosomes = _chromosome_splitter(
                all_chromosomes, leave_out_chromosomes, is_train=is_train
            )

            for chromosome in input_chromosomes:
                idx_peak_list = celltype_peak_annot.index[
                    celltype_peak_annot["Chromosome"] == chromosome
                ].tolist()
                idx_peak_start = idx_peak_list[0]
                idx_peak_end = idx_peak_list[-1]
                for i in range(idx_peak_start, idx_peak_end, step):
                    shift = np.random.randint(-step // 2, step // 2)
                    start_index = max(0, i + shift)
                    end_index = start_index + num_region_per_sample

                    celltype_annot_i = celltype_peak_annot.iloc[
                        start_index:end_index, :
                    ]
                    if celltype_annot_i.shape[0] < num_region_per_sample:
                        logging.debug("Not enough regions in the last batch")
                        continue
                    # if celltype_annot_i.iloc[-1].End - celltype_annot_i.iloc[0].Start > 5000000:
                    #     end_index = celltype_annot_i[celltype_annot_i.End -
                    #                                  celltype_annot_i.Start < 5000000].index[-1]
                    if celltype_annot_i["Start"].min() < 0:
                        continue
                    # add small gaussian noise
                    peak_data_i = peak_data[start_index:end_index]
                    if is_train:
                        peak_data_i = peak_data_i + csr_matrix(
                            np.random.normal(0, 0.0001, peak_data_i.shape)
                        )

                    peak_data_i = coo_matrix(peak_data_i)

                    if not is_pretrain:
                        target_i = coo_matrix(target_data[start_index:end_index])
                        tssidx_i = tssidx_data[start_index:end_index]

                    if peak_data_i.shape[0] == num_region_per_sample:
                        peak_list.append(peak_data_i)
                        cell_list.append(cell_label)
                        if not is_pretrain:
                            target_list.append(target_i)
                            tssidx_list.append(tssidx_i)

        return peak_list, target_list, tssidx_list


class MPRADataset(RegionDataset):
    def __init__(
        self,
        root,
        metadata_path,
        num_region_per_sample,
        mpra_feather_path,
        focus,
        data_type="fetal",
        quantitative_atac=False,
    ):
        super().__init__(
            root,
            metadata_path,
            num_region_per_sample,
            data_type=data_type,
            quantitative_atac=quantitative_atac,
        )
        self.mpra_feather_path = mpra_feather_path
        self.focus = focus
        self.mpra = pd.read_feather(self.mpra_feather_path)
        self.load_mpra_data()

    def load_mpra_data(self):
        # Generate sample list
        self.sample_list = []
        for chr in self.annot.Chromosome.unique():
            idx_sample_list = self.annot.index[self.annot["Chromosome"] == chr].tolist()
            idx_sample_start = idx_sample_list[0]
            idx_sample_end = idx_sample_list[-1]
            for i in range(idx_sample_start, idx_sample_end, 5):
                start_index = i
                end_index = i + self.num_region_per_sample
                self.sample_list.append((start_index, end_index))

        # Pre-sample indices for each MPRA entry
        self.sampled_indices = np.random.choice(
            range(len(self.sample_list)), size=len(self.mpra), replace=True
        )

    def __len__(self):
        return len(self.mpra)

    def __getitem__(self, idx):
        mpra_row = self.mpra.iloc[idx]
        sample_idx = self.sampled_indices[idx]
        start_index, end_index = self.sample_list[sample_idx]

        # Get the original peak data for the sampled region
        c_data = self.peaks[sample_idx].toarray()

        # Insert MPRA data at the focus index
        c_data[self.focus] = mpra_row.values[1:] + c_data[self.focus]
        c_data[c_data > 1] = 1
        c_data[self.focus, 282] = 1

        # Create target data (all zeros for MPRA prediction)
        t_data = np.zeros((self.num_region_per_sample, 2))

        return {
            "region_motif": c_data.astype(np.float32),
            "mask": np.ones(self.num_region_per_sample),
            "exp_label": t_data.astype(np.float32),
        }


class InferenceRegionDataset(RegionDataset):
    """Same as RegionDataset but load the exp.feather to get gene index in peaks
       This is for backward compatibility with old (version 1) datasets. Consider use InferenceRegionMotifDataset for new datasets.

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
        metadata_path = os.path.join(self.root, metadata_path)
        if isinstance(gene_list, str):
            if "," in gene_list:
                gene_list = gene_list.split(",")
            elif os.path.exists(gene_list):
                gene_list = np.loadtxt(gene_list, dtype=str)
        self.gene_list = gene_list if gene_list is not None else []
        self.gencode_obj = gencode_obj
        (
            peaks,
            targets,
            tssidx,
            gene_names,
            strands,
            tss_peaks,
            chromosome,
            peak_coord,
        ) = self._make_dataset(
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
        self.chromosome = chromosome
        self.peak_coord = peak_coord

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
        celltype_metadata = pd.read_csv(celltype_metadata_path, sep=",", dtype=str)
        file_id_list, cell_dict, datatype_dict = self._cell_splitter(
            celltype_metadata,
            leave_out_celltypes,
            datatypes,
            is_train=is_train,
            is_pretrain=is_pretrain,
        )
        peak_list = []
        cell_list = []
        chromosome_list = []
        peak_coord_list = []
        target_list = [] if not is_pretrain else None
        tssidx_list = [] if not is_pretrain else None
        gene_list = []
        strand_list = []
        tss_peak_list = []
        for file_id in file_id_list:
            cell_label = cell_dict[file_id]
            data_type = datatype_dict[file_id]
            logging.debug(file_id, data_path, data_type)
            paths_dict = self._generate_paths(
                file_id, data_path, data_type, quantitative_atac=quantitative_atac
            )

            celltype_peak_annot = pd.read_csv(
                paths_dict["celltype_annot_csv"], sep=","
            ).drop("index", axis=1)

            try:
                peak_data = load_npz(paths_dict["peak_npz"])
                logging.debug(f"Feature shape: {peak_data.shape}")
            except FileNotFoundError:
                logging.debug(f"File not found - FILE ID: {file_id}")
                continue

            if os.path.exists(paths_dict["exp_feather"]):
                exp_df = pd.read_feather(paths_dict["exp_feather"])
            else:
                # construct exp_df from gencode_obj and save it to feather
                # TODO assembly is hardcoded here!!!
                exp_df = self.gencode_obj["hg19"].get_exp_feather(
                    celltype_peak_annot.reset_index()
                )
                exp_df.to_feather(paths_dict["exp_feather"])

            if not is_pretrain:
                target_data = np.load(paths_dict["target_npy"])
                tssidx_data = np.load(paths_dict["tssidx_npy"])
                logging.debug(f"Target shape: {target_data.shape}")
                atac_cutoff = (
                    1
                    - (peak_data[:, 282] >= EXPRESSION_ATAC_CUTOFF).toarray().flatten()
                )
                target_data[atac_cutoff, :] = 0

            if quantitative_atac is False:
                peak_data[:, 282] = 1

            all_chromosomes = celltype_peak_annot["Chromosome"].unique().tolist()
            c = _chromosome_splitter(
                all_chromosomes, leave_out_chromosomes, is_train=is_train
            )

            exp_df = exp_df.query(
                "gene_name.isin(@self.gene_list) & Chromosome.isin(@input_chromosomes)"
            )
            logging.debug(exp_df)
            # instead of loop over chromosome, loop over gene
            for gene, gene_df in exp_df.groupby("gene_name"):
                gene_name = gene_df["gene_name"].values[0]
                chrom = gene_df["Chromosome"].values[0]
                tss_peak = gene_df["index"].values
                strand = 0 if gene_df["Strand"].values[0] == "+" else 1
                idx = (
                    gene_df["index"].values[0]
                    if strand == 0
                    else gene_df["index"].values[-1]
                )

                start_idx = idx - num_region_per_sample // 2
                end_idx = idx + num_region_per_sample // 2
                if start_idx < 0 or end_idx >= peak_data.shape[0]:
                    continue
                celltype_annot_i = celltype_peak_annot.iloc[start_idx:end_idx, :]
                logging.debug(celltype_peak_annot.iloc[tss_peak])
                peak_coord = celltype_annot_i[["Start", "End"]].values
                if celltype_annot_i.shape[0] < num_region_per_sample:
                    continue
                peak_data_i = coo_matrix(peak_data[start_idx:end_idx])

                if not is_pretrain:
                    target_i = coo_matrix(target_data[start_idx:end_idx])
                    tssidx_i = tssidx_data[start_idx:end_idx]

                if peak_data_i.shape[0] == num_region_per_sample:
                    peak_list.append(peak_data_i)
                    cell_list.append(cell_label)
                    if not is_pretrain:
                        target_list.append(target_i)
                        tssidx_list.append(tssidx_i)
                    gene_list.append(gene_name)
                    strand_list.append(strand)
                    tss_peak = tss_peak - start_idx
                    tss_peak = tss_peak[tss_peak < num_region_per_sample]
                    tss_peak_list.append(tss_peak)
                    chromosome_list.append(chrom)
                    peak_coord_list.append(peak_coord)

        return (
            peak_list,
            target_list,
            tssidx_list,
            gene_list,
            strand_list,
            tss_peak_list,
            chromosome_list,
            peak_coord_list,
        )

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
        chromosome = self.chromosome[index]
        peak_coord = self.peak_coord[index]
        tss_peak_mask = np.zeros(self.num_region_per_sample)
        tss_peak_mask[tss_peak] = 1
        if self.mask_ratio > 0:
            mask = np.hstack(
                [
                    np.zeros(
                        int(
                            self.num_region_per_sample
                            - self.num_region_per_sample * self.mask_ratio
                        )
                    ),
                    np.ones(int(self.num_region_per_sample * self.mask_ratio)),
                ]
            )
            np.random.shuffle(mask)
        else:
            mask = tssidx
        if self.transform is not None:
            peak, mask, target = self.transform(peak, tssidx, target)
        if peak.shape[0] == 1:
            peak = peak.squeeze(0)
        item = {
            "region_motif": peak.toarray().astype(np.float32),
            "mask": mask,
            "gene_name": gene_name,
            "tss_peak": tss_peak_mask,
            "chromosome": chromosome,
            "peak_coord": peak_coord,
            "all_tss_peak": np.pad(
                np.unique(tss_peak),
                (0, self.num_region_per_sample - len(np.unique(tss_peak))),
                mode="constant",
                constant_values=-1,
            ),
            "strand": strand,
            "exp_label": target.toarray().astype(np.float32),
        }
        return item


@dataclass
class RegionMotifConfig:
    root: str
    data: str
    celltype: str
    normalize: bool = True
    motif_scaler: float = 1.0
    leave_out_motifs: Optional[str] = None
    drop_zero_atpm: bool = True


class RegionMotif:
    def __init__(self, cfg: RegionMotifConfig):
        self.cfg = cfg
        self.dataset = zarr.open_group(os.path.join(cfg.root, cfg.data), mode="r")
        self.data = self.dataset["data"][:]
        self.peak_names = self.dataset["peak_names"][:]
        self.motif_names = self.dataset["motif_names"][:]
        self.motif_scaler = cfg.motif_scaler
        self.leave_out_motifs = cfg.leave_out_motifs
        self.celltype = cfg.celltype
        self.normalize = cfg.normalize
        self.drop_zero_atpm = cfg.drop_zero_atpm
        if self.leave_out_motifs:
            self.leave_out_motifs = [int(m) for m in self.leave_out_motifs.split(",")]

        self._process_peaks()
        self._load_celltype_data()

    def _process_peaks(self):
        df = pd.DataFrame(self.peak_names[:], columns=["peak_names"])
        df["Chromosome"] = df["peak_names"].apply(lambda x: x.split(":")[0])
        df["Start"] = df["peak_names"].apply(
            lambda x: int(x.split(":")[1].split("-")[0])
        )
        df["End"] = df["peak_names"].apply(lambda x: int(x.split(":")[1].split("-")[1]))
        self._peaks = df.reset_index(drop=True).reset_index()

    def _load_celltype_data(self):
        self.atpm = self.dataset[f"atpm/{self.celltype}"][:]
        if self.drop_zero_atpm:
            atpm_nonzero_idx = np.nonzero(self.atpm)[0]
            self.atpm = self.atpm[atpm_nonzero_idx]
            self.peak_names = self.peak_names[atpm_nonzero_idx]
            self.data = self.data[atpm_nonzero_idx]
            self._peaks = self._peaks.iloc[atpm_nonzero_idx]

        if f"expression_positive/{self.celltype}" in self.dataset:
            self.expression_positive = self.dataset[
                f"expression_positive/{self.celltype}"
            ][:]
            self.expression_negative = self.dataset[
                f"expression_negative/{self.celltype}"
            ][:]

            self.tss = self.dataset[f"tss/{self.celltype}"][:]

            gene_idx_info_index = self.dataset["gene_idx_info_index"][:]
            gene_idx_info_name = self.dataset["gene_idx_info_name"][:]
            gene_idx_info_strand = self.dataset["gene_idx_info_strand"][:]

            if self.drop_zero_atpm:
                self.expression_positive = self.expression_positive[atpm_nonzero_idx]
                self.expression_negative = self.expression_negative[atpm_nonzero_idx]
                self.tss = self.tss[atpm_nonzero_idx]
                gene_idx_info_drop_zero_atpm = []
                idx_to_iloc = {v: i for i, v in enumerate(gene_idx_info_index)}
                for idx_non_zero_atpm, idx in enumerate(atpm_nonzero_idx):
                    if idx in idx_to_iloc:
                        gene_idx_info_drop_zero_atpm.append(
                            (
                                idx_non_zero_atpm,
                                gene_idx_info_name[idx_to_iloc[idx]],
                                gene_idx_info_strand[idx_to_iloc[idx]],
                            )
                        )
                self.gene_idx_info = pd.DataFrame(
                    gene_idx_info_drop_zero_atpm,
                    columns=["index", "gene_name", "strand"],
                )
            else:
                self.gene_idx_info = pd.DataFrame(
                    {
                        "index": gene_idx_info_index,
                        "gene_name": gene_idx_info_name,
                        "strand": gene_idx_info_strand,
                    }
                )
            self.expression_positive[self.atpm < EXPRESSION_ATAC_CUTOFF] = 0
            self.expression_negative[self.atpm < EXPRESSION_ATAC_CUTOFF] = 0

    @property
    def peaks(self):
        return self._peaks

    @property
    def num_peaks(self):
        return len(self.peak_names)

    @property
    def num_motifs(self):
        return len(self.motif_names)

    @property
    def normalized_data(self):
        if not self.normalize:
            return self.data
        if hasattr(self, "_normalized_data"):
            return self._normalized_data
        max_values = self.data.max(axis=0)
        normalized_data = self.data / (max_values * self.motif_scaler)
        normalized_data[normalized_data > 1] = 1
        self._normalized_data = normalized_data
        return normalized_data

    @property
    def normalizing_factor(self):
        if not self.normalize:
            return 1
        max_values = self.data.max(axis=0)
        return max_values * self.motif_scaler

    def __repr__(self):
        return f"RegionMotif(num_peaks={self.num_peaks}, num_motifs={self.num_motifs}, celltype={self.celltype})"



class RegionMotifDataset(Dataset):
    def __init__(
        self,
        zarr_path: str,
        celltypes: str | None = None,
        transform=None,
        quantitative_atac: bool = False,
        sampling_step: int = 50,
        num_region_per_sample: int = 1000,
        leave_out_chromosomes: str | None = None,
        leave_out_celltypes: str | None = None,
        is_train: bool = True,
        mask_ratio: float = 0.0,
        drop_zero_atpm: bool = True,
    ):
        # try split by comma
        if isinstance(zarr_path, str):
            zarr_path = zarr_path.split(",")
        self.zarr_path = zarr_path

        # Get available celltypes from zarr paths by checking atpm group
        self.available_celltypes = []
        for zarr_path in self.zarr_path:
            zarr_root = zarr.open(zarr_path, mode='r')
            if 'atpm' in zarr_root:
                self.available_celltypes.extend(list(zarr_root['atpm'].keys()))
        self.available_celltypes = list(set(self.available_celltypes))

        # Filter celltypes if specified
        if celltypes is not None:
            requested_celltypes = celltypes.split(",")
            self.celltypes = [ct for ct in requested_celltypes if ct in self.available_celltypes]
            if len(self.celltypes) < len(requested_celltypes):
                missing = set(requested_celltypes) - set(self.celltypes)
                logging.warning(f"Some requested celltypes were not found in the zarr files: {missing}")
        else:
            self.celltypes = self.available_celltypes

        self.transform = transform
        self.quantitative_atac = quantitative_atac
        self.sampling_step = sampling_step
        self.num_region_per_sample = num_region_per_sample
        self.leave_out_chromosomes = (
            leave_out_chromosomes if leave_out_chromosomes else []
        )
        self.leave_out_celltypes = (
            leave_out_celltypes.split(",") if leave_out_celltypes else []
        )
        self.is_train = is_train
        self.mask_ratio = mask_ratio
        self.drop_zero_atpm = drop_zero_atpm
        self.region_motifs = self._load_region_motifs()
        self.sample_indices = []
        self.setup()

    def _load_region_motifs(self) -> Dict[str, RegionMotif]:
        region_motifs = {}
        for celltype in self.celltypes:
            if (
                self.is_train
                and len(self.leave_out_celltypes) > 0
                and celltype in self.leave_out_celltypes
            ):
                continue
            if (
                not self.is_train
                and len(self.leave_out_celltypes) > 0
                and celltype not in self.leave_out_celltypes
            ):
                continue
            
            # Find which zarr file contains this celltype
            zarr_path = None
            for path in self.zarr_path:
                if celltype in zarr.open(path, mode='r')['atpm'].keys():
                    zarr_path = path
                    break
                    
            if zarr_path is None:
                logging.warning(f"Could not find zarr file containing celltype {celltype}")
                continue
                    
            cfg = RegionMotifConfig(
                root=os.path.dirname(zarr_path),
                data=os.path.basename(zarr_path),
                celltype=celltype,
                drop_zero_atpm=self.drop_zero_atpm,
            )
            region_motifs[celltype] = RegionMotif(cfg)
        return region_motifs

    def setup(self):
        for celltype, region_motif in tqdm(self.region_motifs.items()):
            peaks = region_motif.peaks
            all_chromosomes = peaks["Chromosome"].unique().tolist()
            input_chromosomes = _chromosome_splitter(
                all_chromosomes, self.leave_out_chromosomes, self.is_train
            )
            for chromosome in input_chromosomes:
                idx_peak_list = peaks.query("Chromosome==@chromosome").index.values
                idx_peak_start = idx_peak_list[0]
                idx_peak_end = idx_peak_list[-1]

                for i in range(idx_peak_start, idx_peak_end, self.sampling_step):
                    shift = np.random.randint(
                        -self.sampling_step // 2, self.sampling_step // 2
                    )
                    start_index = max(0, i + shift)
                    end_index = start_index + self.num_region_per_sample
                    if end_index >= peaks.shape[0]:
                        continue
                    if (
                        peaks.iloc[end_index].End - peaks.iloc[start_index].Start
                        > 5000000
                        or peaks.iloc[end_index].End - peaks.iloc[start_index].Start < 0
                    ):
                        continue
                    if end_index < peaks.shape[0]:
                        self.sample_indices.append((celltype, start_index, end_index))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, index):
        celltype, start_index, end_index = self.sample_indices[index]
        region_motif = self.region_motifs[celltype]

        region_motif_i = region_motif.normalized_data[start_index:end_index]
        peaks_i = region_motif.peaks.iloc[start_index:end_index]
        if hasattr(region_motif, "expression_positive"):
            expression_positive = region_motif.expression_positive[start_index:end_index]
            expression_negative = region_motif.expression_negative[start_index:end_index]
            tss = region_motif.tss[start_index:end_index]
        else:
            logging.warning(
                f"No expression data for {celltype}. Using zeros as placeholder. Don't do expression finetune using this dataset."
            )
            expression_positive = np.zeros(region_motif_i.shape[0])
            expression_negative = np.zeros(region_motif_i.shape[0])
            tss = np.zeros(region_motif_i.shape[0])
        atpm = region_motif.atpm[start_index:end_index]

        if not self.quantitative_atac:
            region_motif_i = np.concatenate(
                [region_motif_i, np.ones((region_motif_i.shape[0], 1))], axis=1
            )
        else:
            normalized_atpm = atpm.reshape(-1, 1) / atpm.max()
            region_motif_i = np.concatenate([region_motif_i, normalized_atpm], axis=1)

        target_data = np.column_stack((expression_positive, expression_negative))
        target_i = coo_matrix(target_data)

        if self.mask_ratio > 0:
            mask = np.random.choice(
                [0, 1], size=len(tss), p=[1 - self.mask_ratio, self.mask_ratio]
            )
        else:
            mask = tss

        if self.transform:
            region_motif_i, mask, target_i = self.transform(
                region_motif_i, mask, target_i
            )

        data = {
            "region_motif": region_motif_i.astype(np.float32),
            "mask": mask,
            "atpm": atpm.reshape(-1, 1),
            "chromosome": peaks_i["Chromosome"].values[0],
            "peak_coord": peaks_i[["Start", "End"]].values,
            "exp_label": target_i.toarray().astype(np.float32),
            "celltype": celltype,
        }

        return data

    def get_item_from_coord(self, chr, start, end, celltype):
        region_motif = self.region_motifs[celltype]
        peaks = region_motif.peaks
        start_index = peaks[
            (peaks["Chromosome"] == chr) & (peaks["Start"] >= start)
        ].index[0]
        end_index = (
            peaks[(peaks["Chromosome"] == chr) & (peaks["End"] <= end)].index[-1] + 1
        )

        return self.__getitem__(
            self.sample_indices.index((celltype, start_index, end_index))
        )

class InferenceRegionMotifDataset(RegionMotifDataset):
    def __init__(self, assembly, gencode_obj, gene_list=None, **kwargs):
        self.gencode_obj = gencode_obj[assembly]
        if isinstance(gene_list, str):
            if "," in gene_list:
                gene_list = gene_list.split(",")
            elif os.path.exists(gene_list):
                gene_list = np.loadtxt(gene_list, dtype=str)
        self.gene_list = (
            gene_list
            if gene_list is not None
            else self.gencode_obj.gtf["gene_name"].unique()
        )
        super().__init__(**kwargs)

    def setup(self):
        """Setup focus on gene list and TSS sites"""
        for celltype, region_motif in tqdm(self.region_motifs.items()):
            if (
                self.is_train
                and len(self.leave_out_celltypes) > 0
                and celltype in self.leave_out_celltypes
            ):
                continue
            if (
                not self.is_train
                and len(self.leave_out_celltypes) > 0
                and celltype not in self.leave_out_celltypes
            ):
                continue

            peaks = region_motif.peaks
            all_chromosomes = peaks["Chromosome"].unique().tolist()
            input_chromosomes = _chromosome_splitter(
                all_chromosomes, self.leave_out_chromosomes, self.is_train
            )
            logging.debug(region_motif.gene_idx_info.drop_duplicates(ignore_index=True))
            # Loop over genes and their TSS sites
            for gene, gene_df in (
                region_motif.gene_idx_info.drop_duplicates(ignore_index=True)
                .query("gene_name.isin(@self.gene_list)")
                .groupby("gene_name")
            ):
                gene_name = gene_df["gene_name"].values[0]
                strand = gene_df["strand"].values[0]
                strand = 0 if strand == "+" else 1
                tss_peaks = gene_df["index"].values
                chrom = region_motif.peaks.iloc[tss_peaks[0]]["Chromosome"]
                # Process each TSS for the gene
                for tss_idx in tss_peaks:
                    start_idx = tss_idx - self.num_region_per_sample // 2
                    end_idx = tss_idx + self.num_region_per_sample // 2

                    if start_idx < 0 or end_idx >= region_motif.peaks.shape[0]:
                        continue

                    peak_coord = region_motif.peaks.iloc[start_idx:end_idx][
                        ["Start", "End"]
                    ].values

                    # Store all necessary information for each TSS
                    self.sample_indices.append(
                        {
                            "celltype": celltype,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "gene_name": gene_name,
                            "strand": strand,
                            "tss_idx": tss_idx,
                            "tss_peaks": tss_peaks
                            - start_idx,  # Convert to relative positions
                            "chromosome": chrom,
                            "peak_coord": peak_coord,
                        }
                    )

    def __getitem__(self, index):
        sample_info = self.sample_indices[index]
        celltype = sample_info["celltype"]
        start_idx = sample_info["start_idx"]
        end_idx = sample_info["end_idx"]
        region_motif = self.region_motifs[celltype]

        # Get the data for this region
        region_motif_i = region_motif.normalized_data[start_idx:end_idx]
        expression_positive = region_motif.expression_positive[start_idx:end_idx]
        expression_negative = region_motif.expression_negative[start_idx:end_idx]
        tss = region_motif.tss[start_idx:end_idx]
        atpm = region_motif.atpm[start_idx:end_idx]

        # Handle ATAC signal
        if not self.quantitative_atac:
            region_motif_i = np.concatenate(
                [region_motif_i, np.ones((region_motif_i.shape[0], 1))], axis=1
            )
        else:
            normalized_atpm = atpm.reshape(-1, 1) / atpm.max()
            region_motif_i = np.concatenate([region_motif_i, normalized_atpm], axis=1)

        # Prepare target data
        target_data = np.column_stack((expression_positive, expression_negative))
        target_i = coo_matrix(target_data)

        # Handle masking
        if self.mask_ratio > 0:
            mask = np.random.choice(
                [0, 1], size=len(tss), p=[1 - self.mask_ratio, self.mask_ratio]
            )
        else:
            mask = tss

        if self.transform:
            region_motif_i, mask, target_i = self.transform(
                region_motif_i, mask, target_i
            )

        # Filter and pad TSS peaks
        valid_tss_peaks = sample_info["tss_peaks"]
        valid_tss_peaks = valid_tss_peaks[
            (valid_tss_peaks >= 0) & (valid_tss_peaks < self.num_region_per_sample)
        ]
        padded_tss_peaks = np.pad(
            valid_tss_peaks,
            (0, self.num_region_per_sample - len(valid_tss_peaks)),
            mode="constant",
            constant_values=-1,
        )

        return {
            "region_motif": region_motif_i.astype(np.float32),
            "mask": mask,
            "gene_name": sample_info["gene_name"],
            "tss_peak": sample_info["tss_idx"] - start_idx,  # Current TSS position
            "chromosome": sample_info["chromosome"],
            "peak_coord": sample_info["peak_coord"],
            "all_tss_peak": padded_tss_peaks,  # All TSS positions for this gene
            "strand": sample_info["strand"],
            "exp_label": target_i.toarray().astype(np.float32),
            "celltype": celltype,
        }
