import yaml
from pathlib import Path


class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load()

    def _load(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        return self._config.get(key, default)

    @property
    def data(self):
        return self._config