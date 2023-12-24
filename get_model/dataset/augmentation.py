from torchvision import transforms
import torch
import numpy as np
import scipy




# class ATACSample(object):
#     """Object contains peak motif vector and peak sequence of a single sample."""
#     def __init__(self, peak_motif, peak_sequence=None):
#         self.peak_motif = peak_motif
#         self.peak_sequence = peak_sequence

#     def __repr__(self):
#         return f'ATACSample(Number of peaks: {self.peak_motif.shape[0]})'
    
#     def __len__(self):
#         return self.peak_motif.shape[0]


# class PeaksSequence(object):
#     """A tuple containing a concatenated sequence and segmentation idx for NUM_PEAKS peaks."""

#     def __init__(self, sequence, region):
#         """
#         Initialize PeaksSequence.

#         Args:
#             sequence: The concatenated sequence.
#             region: The region containing segmentation indices.
#         """
#         self.sequence = sequence # (N,4) N = num_region_per_sample * region_len
#         self.starts = region["SeqStartIdx"].tolist() - region["SeqStartIdx"].min()
#         self.ends = region["SeqEndIdx"].tolist() - region["SeqStartIdx"].min()
#         self.region = region

#     def __getitem__(self, index):
#         start = self.starts[index]
#         end = self.ends[index]
#         return self.sequence[start:end]

#     def __len__(self):
#         return len(self.starts)

#     def __repr__(self):
#         return self.region.__repr__()
    

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.num_regions = input_size
        self.num_mask = int(mask_ratio * self.num_regions)

    def __repr__(self):
        repr_str = "Maks: total regions {}, mask regions {}".format(
            self.num_regions, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack(
            [
                np.zeros(self.num_regions - self.num_mask),
                np.ones(self.num_mask),
            ]
        )
        np.random.shuffle(mask)
        return mask

class TSSMaskingGenerator:
    def __init__(self, mask_tss=True):
        self.mask_tss = mask_tss

    def __repr__(self):
        repr_str = "Mask: TSS masking"
        return repr_str

    def __call__(self, tss_idx): # tss_idx: nd.array((200,2))
        tss_pos = (tss_idx.sum(1))>0
        return tss_pos


# class SequenceGenerator(object):
#     """
#     A callable class that samples sequences from a given peaks sequence. The sampled sequences are centered around the peak
#     and have a target sequence length. If the length of the sequence is less than the target sequence length, it is padded
#     with zeros. The class takes a peaks sequence as input and returns a numpy array of sampled sequences.

#     Args:
#     target_sequence_length (int): The target length of the sampled sequences.

#     Returns:
#     numpy.ndarray: A numpy array of sampled sequences.
#     """

#     def __init__(self, target_sequence_length=2000, shift=100):
#         self.target_sequence_length = target_sequence_length
#         self.shift = shift

#     def __call__(self, peaks_sequence: PeaksSequence):
#         results = []
#         for i in range(len(peaks_sequence)):
#             sequence = peaks_sequence[i].toarray()
#             shift = np.random.randint(-100, 100)
#             sequence_len = sequence.shape[0]
#             if sequence_len >= self.target_sequence_length:
#                 center = sequence_len // 2
#                 start = max(0, center - self.target_sequence_length // 2 + shift)
#                 end = min(start + self.target_sequence_length, sequence_len)
#                 sequence_i = sequence[start:end]
#             else:
#                 sequence_i = sequence
#             sequence_len = sequence_i.shape[0]
#             high = self.target_sequence_length - sequence_len
#             if high <= 0:
#                 pad_left = 0
#             else:
#                 pad_left = np.random.randint(0, high)
#             pad_right = self.target_sequence_length - sequence_len - pad_left
#             # pad axis 0 
#             sequence_i = np.pad(sequence_i, ((pad_left, pad_right), (0, 0)), "constant", constant_values=(0, 0))
#             results.append(sequence_i)
#         return np.stack(results, axis=0)

#     def __repr__(self):
#         repr_str = "Target sequence length: {}bp\nShift size: {}bp".format(
#             self.target_sequence_length, self.shift
#         )
#         return repr_str


class DataAugmentationForGETPeak(object):
    def __init__(self, args):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.masked_position_generator = RandomMaskingGenerator(
            args.num_region_per_sample, args.mask_ratio
        )

    def __call__(self, region, sequence):
        if sequence.sum() != 0:
            sequence = torch.Tensor(sequence)
        if isinstance(region, scipy.sparse.coo_matrix):
            region = region.toarray()
        region = self.transform(region)
        mask = self.masked_position_generator()
        return region, sequence, mask

    def __repr__(self):
        repr = "(DataAugmentationForGETPeak,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr



class DataAugmentationForGETPeakFinetune(object):
    def __init__(self, args):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.masked_position_generator = TSSMaskingGenerator(args.mask_tss)

    def __call__(self, region, sequence, tss_idx, target):
        if sequence.sum() != 0:
            sequence = torch.Tensor(sequence)
        if isinstance(region, scipy.sparse.coo_matrix):
            region = region.toarray()
        region = self.transform(region)
        mask = self.masked_position_generator(tss_idx)
        target = torch.Tensor(target.toarray())
        return region, sequence, mask, target

    def __repr__(self):
        repr = "(DataAugmentationForGETPeakFinetune,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr

