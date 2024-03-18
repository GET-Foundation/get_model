# Registry of data modalities, will be used in dataset, model and trainer. The point is to make the code more modular and easy to maintain. For example, some model may only use sequence data, some may use both sequence and signal data, etc. this requires different dataset getitem function outputs, different collate_fn, and different loss function and different metrics and different trainer function. Ideally, all these should be configurable, and the code should be able to handle different combination of modalities. This is the purpose of this registry.



class Modality:
    """
    Base class for representing a modality in the codebase.

    This class provides a foundation for defining modality-specific configurations,
    transforms, dataset handling, model forward pass, and engine processing.

    """

    def __init__(self, name):
        """
        Initialize a Modality instance.

        Args:
            name (str): The name of the modality.
        """
        self.name = name
        self.config = None

    def get_config(self):
        """
        Get the configuration for the modality.

        Returns:
            dict: The configuration for the modality.
        """
        return self.config
    
    def get_transforms(self, transform_name):
        """
        Get the transforms for the modality.

        Args:
            transform_name (str): The name of the transform.

        Returns:
            object: The transform object.
        """
        return self.config.transform.get(transform_name)