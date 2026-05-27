import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Loads and returns the configuration from a YAML file.

    Args:
        config_path (str, optional): The path to the YAML configuration file.
            If None, resolves to the default configs/config.yaml from the project root.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file cannot be found.
    """
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "configs", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config
