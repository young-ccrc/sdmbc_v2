import os
from os.path import dirname, exists, expanduser, join

import yaml  # type: ignore


class Config:
    def __init__(self):
        # Define potential paths for the configuration
        self.env_config_path = join(
            os.environ.get("VIRTUAL_ENV", ""), ".sdmbc_v2", "config.yaml"
        )
        self.user_config_path = join(expanduser("~"), ".sdmbc_v2", "config.yaml")
        self.global_config_path = "/.sdmbc_v2/config.yaml"
        self.default_config_path = join(
            dirname(dirname(os.path.realpath(__file__))), "config.yaml"
        )

        # Load the configuration
        self.config = self.load_config()

    def load_config(self):
        # Check each path and load the first found
        for path in [
            self.env_config_path,
            self.user_config_path,
            self.global_config_path,
            self.default_config_path,
        ]:
            if exists(path):
                with open(path, "r") as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError("No configuration file found.")

    def __getattr__(self, item):
        # Allow dynamic access to configuration items as attributes
        return self.config.get(item)


# Create a single global instance of Config
config = Config()
