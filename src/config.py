import os
from os.path import dirname, exists, expanduser, join
import yaml  # type: ignore


class Config:
    def __init__(self, path=None):
        """
        Initialize configuration from a specified path, or fallback to standard search paths.
        """
        self.custom_path = path
        self.env_config_path = join(
            os.environ.get("VIRTUAL_ENV", ""), ".sdmbc_v2", "config.yaml"
        )
        self.user_config_path = join(expanduser("~"), ".sdmbc_v2", "config.yaml")
        self.global_config_path = "/.sdmbc_v2/config.yaml"
        self.default_config_path = join(
            dirname(dirname(os.path.realpath(__file__))), "config.yaml"
        )

        self.config = self.load_config()

    def load_config(self):
        if self.custom_path:
            if exists(self.custom_path):
                with open(self.custom_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                raise FileNotFoundError(
                    f"Specified config file not found: {self.custom_path}"
                )

        for path in [
            self.env_config_path,
            self.user_config_path,
            self.global_config_path,
            self.default_config_path,
        ]:
            if exists(path):
                with open(path, "r") as f:
                    return yaml.safe_load(f)

        raise FileNotFoundError("No configuration file found in any known path.")

    def __getattr__(self, item):
        return self.config.get(item)


# Optional: preserve current behavior
config = Config()
