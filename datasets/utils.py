import torch
from torch.utils.data import Dataset

def create_random_subset(dataset: Dataset, subset_size: int):
    """
    Create a random subset of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The input dataset.
        subset_size (int): The desired size of the random subset.

    Returns:
        torch.utils.data.Dataset: The random subset of the dataset.

    """
    # Get the total size of the dataset
    dataset_size = len(dataset)

    # Create a random subset using torch.utils.data.random_split
    random_indices = torch.randperm(dataset_size)[:subset_size]
    random_subset = torch.utils.data.Subset(dataset, random_indices)

    return random_subset


def add_spike_in_dataset(main_dataset: Dataset, spike_dataset: Dataset, spike_ratio: float):
    """
    Add spike-in samples to the main dataset and produce a new dataset.

    Args:
        main_dataset (torch.utils.data.Dataset): The main dataset.
        spike_dataset (torch.utils.data.Dataset): The spike-in dataset.
        spike_ratio (float): The ratio of spike-in samples to add.

    Returns:
        torch.utils.data.Dataset: The new dataset after adding the spike-in samples.

    """
    spike_num = int(len(main_dataset) * spike_ratio)
    # randomly select spike-in samples
    spike_dataset = create_random_subset(spike_dataset, spike_num)
    # add spike-in samples to the main dataset
    main_dataset.extend(spike_dataset)
    return main_dataset