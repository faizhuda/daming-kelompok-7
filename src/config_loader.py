import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
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
        path = Path(__file__).parent.parent / "configs" / "config.yaml"
    else:
        path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    logger.info("load_config: loaded from %s", path)
    return config
