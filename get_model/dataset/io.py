import os
import os.path

import numpy as np
import pandas as pd
from pyranges import PyRanges as pr


def generate_paths(file_id: int, data_path: str, data_type: str, quantitative_atac: bool = False) -> dict:
    """
    Generate a dictionary of paths based on the given parameters.

    Args:
        file_id (int): File ID.
        data_path (str): Path to the data directory.
        data_type (str): Data type.
        quantitative_atac (bool, optional): Specify if natac files should be used. Defaults to False.

    Returns:
        dict: Dictionary of paths with file IDs as keys and corresponding paths as values.

    Raises:
        FileNotFoundError: If the peak file is not found.

    """
    paths_dict = {}
    if quantitative_atac:
        peak_npz_path = os.path.join(
            data_path, data_type, str(file_id) + ".watac.npz")
    else:
        peak_npz_path = os.path.join(
            data_path, data_type, str(file_id) + ".natac.npz")

    if not os.path.exists(peak_npz_path):
        raise FileNotFoundError(
            "Peak file not found: {}".format(peak_npz_path))

    target_npy_path = os.path.join(
        data_path, data_type, str(file_id) + ".exp.npy")
    tssidx_npy_path = os.path.join(
        data_path, data_type, str(file_id) + ".tss.npy")
    seq_path = os.path.join(data_path, data_type,
                            str(file_id) + ".seq.zarr.zip")
    celltype_annot = os.path.join(data_path, data_type, str(file_id) + ".csv")

    paths_dict = {
        "file_id": file_id,
        "peak_npz": peak_npz_path,
        "target_npy": target_npy_path,
        "tssidx_npy": tssidx_npy_path,
        "seq_npz": seq_path,
        "celltype_annot_csv": celltype_annot
    }
    return paths_dict


def prepare_sequence_idx(celltype_annot: pd.DataFrame, slop: int = 100) -> pd.DataFrame:
    """
    Prepares sequence indices for the celltype annotation data.

    Args:
        celltype_annot (pd.DataFrame): Celltype annotation data.
        slop (int, optional): Extended sequence size on both sides per sample. Defaults to 100.

    Returns:
        pd.DataFrame: Updated celltype annotation data with sequence indices.
    """
    celltype_annot["SeqLength"] = (
        celltype_annot["End"] - celltype_annot["Start"] + slop * 2
    )
    celltype_annot["SeqLengthCumulative"] = celltype_annot["SeqLength"].cumsum()
    celltype_annot["SeqStartIdx"] = [0] + celltype_annot[
        "SeqLengthCumulative"
    ].tolist()[:-1]
    celltype_annot["SeqEndIdx"] = celltype_annot["SeqLengthCumulative"].tolist()
    return celltype_annot


def get_ctcf_pos(celltype_annot: pd.DataFrame, ctcf: pd.DataFrame) -> np.ndarray:
    """
    Segment the regions by CTCF binding sites
    """
    celltype_annot_w_ctcf = pr(celltype_annot).join(
        pr(pd.concat((celltype_annot, ctcf)))
    )
    ctcf_pos = []
    x = 0
    for j in celltype_annot_w_ctcf.df["Unnamed: 0_b"].values:
        if np.isnan(j):
            x += 1
        else:
            ctcf_pos.append(x)
    try:
        assert len(ctcf_pos) == celltype_annot.shape[0]
    except AssertionError:
        print(
            f"ctcf_pos length {len(ctcf_pos)} not equal to celltype_annot length {celltype_annot.shape[0]}")
    return np.array(ctcf_pos)


def get_hierachical_ctcf_pos(
    celltype_annot: pd.DataFrame, ctcf: pd.DataFrame, cut=[50, 100, 200]
) -> np.ndarray:
    """
    Segment the regions by multi-level CTCF binding sites, filter ctcf by num_celltype with cut
    """
    multilevel_ctcf_pos = []
    for c in cut:
        ctcf_filtered = ctcf.query("num_celltype>=" + str(c))
        multilevel_ctcf_pos.append(get_ctcf_pos(celltype_annot, ctcf_filtered))
    multilevel_ctcf_pos = np.vstack(multilevel_ctcf_pos).T
    return multilevel_ctcf_pos
