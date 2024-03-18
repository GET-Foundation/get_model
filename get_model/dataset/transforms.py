import numpy as np
import torch
import os
from typing import Any, Callable, List, Optional, Tuple
from typing import Mapping, Tuple, List, Optional, Dict, Sequence
import torch.utils.data as data
import numpy as np
from functools import reduce, wraps
import torch


FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    features: Sequence[str],
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.
    Borrowed from OpenFold codebase.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    # torch generates warnings if feature is already a torch Tensor
    to_tensor = lambda t: torch.tensor(t) if type(t) != torch.Tensor else t.clone().detach()
    tensor_dict = {
        k: to_tensor(v) for k, v in np_example.items() if k in features
    }

    return tensor_dict

def curry1(f):
    """Supply all arguments but the first."""
    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def rev_comp(seq, signal, prob=0.5):
    """
    Reverse complement the sequence and signal.
    Assume the sequence is one-hot encoded as 2D numpy array with shape (L, 4)
    And the signal is 1D numpy array with shape (L,)
    """
    if np.random.rand() < prob:
        return seq, signal
    seq = seq[::-1, ::-1]
    signal = signal[::-1]
    return seq, signal


def normalize_peak_signal_track(peak_signal_track, libsize):
    """
    Normalize peak signal track by library size.

    Args:
        peak_signal_track (torch.Tensor): Peak signal track tensor.
        libsize (float): Library size for normalization.

    Returns:
        torch.Tensor: Normalized peak signal track tensor.
    """
    return peak_signal_track / libsize * 100000000

def resize_peak_signal_track_sequence(peak_signal_track, peak_sequence, sample_len_max):
    """
    Resize peak signal track and peak sequence to the maximum sample length.

    Args:
        peak_signal_track (torch.Tensor): Peak signal track tensor.
        peak_sequence (torch.Tensor): Peak sequence tensor.
        sample_len_max (int): Maximum sample length.

    Returns:
        tuple: Resized peak signal track and peak sequence tensors.
    """
    peak_signal_track.resize_((sample_len_max, peak_signal_track.shape[1]))
    peak_sequence.resize_((sample_len_max, peak_sequence.shape[1]))
    return peak_signal_track, peak_sequence

def rev_comp_transform(peak_sequence, peak_signal_track, prob=0.5):
    """
    Apply reverse complement transformation to peak sequence and signal track with a given probability.

    Args:
        peak_sequence (torch.Tensor): Peak sequence tensor.
        peak_signal_track (torch.Tensor): Peak signal track tensor.
        prob (float, optional): Probability of applying the transformation. Defaults to 0.5.

    Returns:
        tuple: Transformed peak sequence and signal track tensors.
    """
    return rev_comp(peak_sequence, peak_signal_track, prob=prob)

def convolve_peak_signal_track(peak_signal_track, conv=50):
    """
    Apply convolution to the peak signal track.

    Args:
        peak_signal_track (torch.Tensor): Peak signal track tensor.
        conv (int, optional): Convolution window size. Defaults to 50.

    Returns:
        torch.Tensor: Convolved peak signal track tensor.
    """
    return np.convolve(np.array(peak_signal_track).reshape(-1), np.ones(conv)/conv, mode='same')

def threshold_peak_labels(peak_labels, threshold=0.1):
    """
    Threshold peak labels based on aTPM value.

    Args:
        peak_labels (torch.Tensor): Peak labels tensor.
        threshold (float, optional): aTPM threshold value. Defaults to 0.1.

    Returns:
        torch.Tensor: Thresholded peak labels tensor.
    """
    n_peak_labels = peak_labels.shape[-1]
    if n_peak_labels >= 3:
        peak_labels = peak_labels.reshape(-1, n_peak_labels)
        peak_labels[peak_labels[:,2]<threshold, 0] = 0
        peak_labels[peak_labels[:,2]<threshold, 1] = 0
        peak_labels = peak_labels.reshape(-1, n_peak_labels)
    return peak_labels  






############################################
def pad_celltype_peaks(celltype_peaks, n_peak_max):
    """
    Pad celltype peaks to the maximum number of peaks in the batch.

    Args:
        celltype_peaks (list): List of celltype peaks tensors.
        n_peak_max (int): Maximum number of peaks in the batch.

    Returns:
        torch.Tensor: Padded celltype peaks tensor.
    """
    for i in range(len(celltype_peaks)):
        celltype_peaks[i] = np.pad(celltype_peaks[i], ((0, n_peak_max - len(celltype_peaks[i])), (0,0)))
    return np.stack(celltype_peaks, axis=0)

def create_chunk_size(padded_peak_len, tail_len, n_peaks):
    """
    Create chunk sizes based on padded peak lengths, tail lengths, and number of peaks.

    Args:
        padded_peak_len (torch.Tensor): Padded peak length tensor.
        tail_len (torch.Tensor): Tail length tensor.
        n_peaks (torch.Tensor): Number of peaks tensor.

    Returns:
        list: List of chunk sizes.
    """
    return torch.cat([torch.cat([padded_peak_len[i][0:n],tail_len[i].unsqueeze(0)]) for i, n in enumerate(n_peaks)]).tolist()

def create_mask(n_peaks, max_n_peaks, mask_ratio):
    """
    Create a mask tensor based on the number of peaks, maximum number of peaks, and mask ratio.

    Args:
        n_peaks (torch.Tensor): Number of peaks tensor.
        max_n_peaks (int): Maximum number of peaks in the batch.
        mask_ratio (float): Ratio of peaks to mask.

    Returns:
        torch.Tensor: Mask tensor.
    """
    mask = torch.stack([torch.cat([torch.zeros(i), torch.zeros(max_n_peaks-i)-10000]) for i in n_peaks.tolist()])
    maskable_pos = (mask+10000).nonzero()
    batch_size = len(n_peaks)
    for i in range(batch_size):
        maskable_pos_i = maskable_pos[maskable_pos[:,0]==i,1]
        idx = np.random.choice(maskable_pos_i, size=np.ceil(mask_ratio*len(maskable_pos_i)).astype(int), replace=False)
        mask[i,idx] = 1
    return mask

def pad_peak_labels(peak_labels, max_n_peaks):
    """
    Pad peak labels to the maximum number of peaks in the batch.

    Args:
        peak_labels (list): List of peak labels tensors.
        max_n_peaks (int): Maximum number of peaks in the batch.

    Returns:
        torch.Tensor: Padded peak labels tensor.
    """
    for i in range(len(peak_labels)):
        peak_labels[i] = np.pad(peak_labels[i], ((0, max_n_peaks - len(peak_labels[i])), (0,0)))
    return np.stack(peak_labels, axis=0)