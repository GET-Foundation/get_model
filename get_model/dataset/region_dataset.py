import seaborn as sns
from get_model.model.model_refactored import GETRegionFinetune
import os
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, load_npz
from torch.utils.data import Dataset


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
        use_natac: bool = True,
        sampling_step: int = 100,
    ) -> None:
        super(RegionDataset, self).__init__()

        self.root = root
        self.transform = transform
        self.is_train = is_train
        self.leave_out_celltypes = leave_out_celltypes
        self.leave_out_chromosomes = leave_out_chromosomes
        self.use_natac = use_natac
        self.sampling_step = sampling_step

        metadata_path = os.path.join(
            self.root, metadata_path
        )
        peaks, cells, targets, tssidx = self._make_dataset(
            False,
            data_type,
            self.root,
            metadata_path,
            num_region_per_sample,
            leave_out_celltypes,
            leave_out_chromosomes,
            use_natac,
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
Use natac: {self.use_natac}
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
            celltype_list = [
                cell for cell in celltype_list if cell not in leave_out_celltypes]
            print(f"Train cell types list: {celltype_list}")
            print(f"Train data types list: {datatypes}")
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
    def _chromosome_splitter(all_chromosomes, leave_out_chromosomes, is_train=True):
        input_chromosomes = all_chromosomes.copy()
        leave_out_chromosomes = leave_out_chromosomes.split(",")

        if is_train:
            input_chromosomes = [
                chrom for chrom in input_chromosomes if chrom not in leave_out_chromosomes]
        else:
            input_chromosomes = all_chromosomes if leave_out_chromosomes == [
            ] else leave_out_chromosomes

        if isinstance(input_chromosomes, str):
            input_chromosomes = [input_chromosomes]

        print(f"Input chromosomes: {input_chromosomes}")
        return input_chromosomes

    @staticmethod
    def _generate_paths(file_id: int, data_path: str, data_type: str, use_natac: bool = True) -> dict:
        """
        Generate a dictionary of paths based on the given parameters.

        Args:
            file_id (int): File ID.
            data_path (str): Path to the data directory.
            data_type (str): Data type.
            use_natac (bool, optional): Specify if natac files should be used. Defaults to True.

        Returns:
            dict: Dictionary of paths with file IDs as keys and corresponding paths as values.

        Raises:
            FileNotFoundError: If the peak file is not found.
        """
        peak_npz_path = os.path.join(
            data_path, data_type, f"{file_id}.{'natac' if use_natac else 'watac'}.npz"
        )

        if not os.path.exists(peak_npz_path):
            raise FileNotFoundError(f"Peak file not found: {peak_npz_path}")

        target_npy_path = os.path.join(
            data_path, data_type, f"{file_id}.exp.npy")
        tssidx_npy_path = os.path.join(
            data_path, data_type, f"{file_id}.tss.npy")
        celltype_annot = os.path.join(
            data_path, data_type, f"{file_id}.csv.gz")

        return {
            "file_id": file_id,
            "peak_npz": peak_npz_path,
            "target_npy": target_npy_path,
            "tssidx_npy": tssidx_npy_path,
            "celltype_annot_csv": celltype_annot,
        }

    @staticmethod
    def _make_dataset(
        is_pretrain: bool,
        datatypes: str,
        data_path: str,
        celltype_metadata_path: str,
        num_region_per_sample: int,
        leave_out_celltypes: str,
        leave_out_chromosomes: str,
        use_natac: bool,
        is_train: bool,
        step: int = 200,
    ) -> Tuple[List[coo_matrix], List[str], List[coo_matrix], List[np.ndarray]]:
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
            use_natac (bool): Whether to use peak data with no ATAC count values.
            is_train (bool): Whether it is a training dataset.
            step (int, optional): Step size for generating samples. Defaults to 200.

        Returns:
            Tuple[List[coo_matrix], List[str], List[coo_matrix], List[np.ndarray]]: Tuple containing the generated peak data,
            cell labels, target data, and TSS indices.
        """
        celltype_metadata = pd.read_csv(
            celltype_metadata_path, sep=",", dtype=str)
        file_id_list, cell_dict, datatype_dict = RegionDataset._cell_splitter(
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
            paths_dict = RegionDataset._generate_paths(
                file_id, data_path, data_type, use_natac=use_natac
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

            all_chromosomes = celltype_peak_annot["Chromosome"].unique(
            ).tolist()
            input_chromosomes = RegionDataset._chromosome_splitter(
                all_chromosomes, leave_out_chromosomes, is_train=is_train
            )

            for chromosome in input_chromosomes:
                idx_peak_list = celltype_peak_annot.index[celltype_peak_annot["Chromosome"] == chromosome].tolist(
                )
                idx_peak_start = idx_peak_list[0]
                idx_peak_end = idx_peak_list[-1]
                print(idx_peak_end)
                for i in range(idx_peak_start, idx_peak_end, step):
                    shift = np.random.randint(-step // 2, step // 2)
                    start_index = max(0, i + shift)
                    end_index = start_index + num_region_per_sample

                    celltype_annot_i = celltype_peak_annot.iloc[start_index:end_index, :]
                    if celltype_annot_i.iloc[-1].End - celltype_annot_i.iloc[0].Start > 5000000:
                        end_index = celltype_annot_i[celltype_annot_i.End -
                                                     celltype_annot_i.Start < 5000000].index[-1]
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

        return peak_list, cell_list, target_list, tssidx_list


# %%
rd = RegionDataset(root='/home/xf2217/Projects/new_finetune_data_all',
                   metadata_path='cell_type_align.txt',
                   num_region_per_sample=900,
                   is_train=False,
                   transform=None,
                   data_type='fetal',
                   leave_out_celltypes='Astrocytes',
                   leave_out_chromosomes='chr19',
                   use_natac=False,
                   sampling_step=100)

# %%
a = rd.__getitem__(100)
# %%
a['region_motif'].shape
# %%
sns.scatterplot(x=a['region_motif'][:, -1], y=a['exp_label'].mean(1))
# %%
