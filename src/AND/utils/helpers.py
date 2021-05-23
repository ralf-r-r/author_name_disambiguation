import os
import yaml
from pathlib import Path
import AND.utils.logger as logger

__all__ = ['read_configurations']

def read_configurations(config_file_name: str = "config.yaml") -> dict:
    """
    reads the config file
    :param config_file_name: str
    :return: dict
    """
    prefix = Path(os.path.abspath(os.path.realpath(__file__))).parents[3]
    try:
        with open(os.path.join(prefix, "config_files", config_file_name)) as file:
            configuration_file = yaml.full_load(file)
    except FileNotFoundError:
        logger.logging.error(
            "Config file could not be found in /config_files/.. "
        )
        raise FileNotFoundError
    logger.logging.info(
        ">>> Loaded config file successfully"
    )
    return configuration_file
