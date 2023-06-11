from torchvision import transforms
from . import PeaksSequence
import numpy as np

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


class SequenceGenerator(object):
    """
    A callable class that samples sequences from a given peaks sequence. The sampled sequences are centered around the peak
    and have a target sequence length. If the length of the sequence is less than the target sequence length, it is padded
    with zeros. The class takes a peaks sequence as input and returns a numpy array of sampled sequences.

    Args:
    target_sequence_length (int): The target length of the sampled sequences.

    Returns:
    numpy.ndarray: A numpy array of sampled sequences.
    """

    def __init__(self, target_sequence_length=2000, shift=100):
        self.target_sequence_length = target_sequence_length
        self.shift = shift

    def __call__(self, peaks_sequence: PeaksSequence):
        results = []
        for i in range(len(peaks_sequence)):
            sequence = peaks_sequence[i]
            shift = np.random.randint(-100, 100)
            if len(sequence) >= self.target_sequence_length:
                center = len(sequence) // 2
                start = max(0, center - self.target_sequence_length // 2 + shift)
                end = min(start + self.target_sequence_length, len(sequence))
                sequence_i = sequence[start:end]
            else:
                sequence_i = sequence

            pad_left = np.random.randint(0, self.target_sequence_length - len(sequence))
            pad_right = self.target_sequence_length - len(sequence) - pad_left
            sequence_i = np.pad(
                sequence_i, (pad_left, pad_right), "constant", constant_values=0
            )
            results.append(sequence_i)
        return np.stack(results, axis=0)

    def __repr__(self):
        repr_str = "Target sequence length: {}bp\nShift size: {}bp".format(
            self.target_sequence_length, self.shift
        )
        return repr_str


class DataAugmentationForGETPeak(object):
    def __init__(self, args):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.masked_position_generator = RandomMaskingGenerator(
            args.region_size, args.mask_ratio
        )

    def __call__(self, region):
        return self.transform(region), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForGeneFormer,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr


class DataAugmentationForGETSequence(object):
    def __init__(self, args):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.sequence_sampler = SequenceGenerator(args.target_sequence_length, args.shift)
        self.masked_position_generator = RandomMaskingGenerator(
            args.region_size, args.mask_ratio
        )

    def __call__(self, region, peaks_sequence):
        return (
            self.transform(region),
            self.sequence_sampler(peaks_sequence),
            self.masked_position_generator(),
        )

    def __repr__(self):
        repr = "(DataAugmentationForGeneFormer,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(
            self.masked_position_generator
        )
        repr += ")"
        return repr
