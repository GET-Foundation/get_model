import os

import yaml
from pkg_resources import resource_filename


class Config:
    def __init__(self, config_name):
        config_file = self._get_config_file_path(config_name)
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
            # set all config keys as attributes
            for key, value in self.config.items():
                setattr(self, key, value)

    def get(self, key):
        return self.config.get(key)

    @staticmethod
    def _get_config_file_path(config_name):
        """
        Get the full path of the config file

        Parameters
        ----------
        config_name : str
            Name of the config file

        Returns
        -------
        str
            Full path of the config file
        """
        config_path = 'config'
        config_file = f"{config_name}.yaml"
        full_path = resource_filename(
            'caesar', os.path.join(config_path, config_file))

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Config file {config_file} not found in {config_path}")

        return full_path

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if value is not None:
                setattr(self, key, value)
