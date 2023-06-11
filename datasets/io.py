import os
import os.path

import numpy as np
import pandas as pd
from pyranges import PyRanges as pr


def parse_meme_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []
    current_motif = None
    alphabet = lines[2].split()[1]
    # skip the first 10 lines
    lines = lines[9:]
    for i, line in enumerate(lines):
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
            current_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif current_motif is not None and line.startswith("letter-probability matrix"):
            letter_prob_matrix = []
            current_motif["width"] = int(line.split()[5])
            for j in range(current_motif["width"]):
                row = list(map(float, lines[i + j + 1].split()))
                letter_prob_matrix.append(row)
            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            current_motif["width"] = len(letter_prob_matrix)
            if current_motif["width"] < 29:
                current_motif["letter_prob_matrix"] = np.concatenate(
                    (
                        current_motif["letter_prob_matrix"],
                        np.zeros((29 - current_motif["width"], 4)),
                    ),
                    axis=0,
                )
    if current_motif is not None:
        # pad to 29, 4
        motifs.append(current_motif)

    motifs = np.stack([motif["letter_prob_matrix"] for motif in motifs], axis=0)
    return motifs


def generate_paths(file_id: int, data_path: str, data_type: str, use_natac: bool = True) -> dict:
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
    paths_dict = {}
    if use_natac:
        peak_npz_path = os.path.join(data_path, data_type, str(file_id) + ".natac.npz")
    else:
        peak_npz_path = os.path.join(data_path, data_type, str(file_id) + ".watac.npz")

    if not os.path.exists(peak_npz_path):
        raise FileNotFoundError("Peak file not found: {}".format(peak_npz_path))

    target_npy_path = os.path.join(data_path, data_type, str(file_id) + ".exp.npy")
    tssidx_npy_path = os.path.join(data_path, data_type, str(file_id) + ".tss.npy")
    seq_npz_path = os.path.join(data_path, data_type, str(file_id) + ".seq.slop_100.npz")
    celltype_annot = os.path.join(data_path, data_type, file_id + ".csv")

    paths_dict[file_id] = {
        "peak_npz": peak_npz_path,
        "target_npy": target_npy_path,
        "tssidx_npy": tssidx_npy_path,
        "seq_npz": seq_npz_path,
        "celltype_annot_csv": celltype_annot
    }
    return paths_dict


def prepare_sequence_idx(celltype_annot: pd.DataFrame, num_region_per_sample: int = 200) -> pd.DataFrame:
    """
    Prepares sequence indices for the celltype annotation data.

    Args:
        celltype_annot (pd.DataFrame): Celltype annotation data.
        num_region_per_sample (int, optional): Number of regions per sample. Defaults to 200.

    Returns:
        pd.DataFrame: Updated celltype annotation data with sequence indices.
    """
    celltype_annot["SeqLength"] = (
        celltype_annot["End"] - celltype_annot["Start"] + num_region_per_sample
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
    assert len(ctcf_pos) == celltype_annot.shape[0]
    return np.array(ctcf_pos)


def get_hierachical_ctcf_pos(
    celltype_annot: pd.DataFrame, ctcf: pd.DataFrame, cut=[10, 20, 50, 100, 200]
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
