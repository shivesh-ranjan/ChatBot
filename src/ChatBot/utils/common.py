import os
from box.exceptions import BoxValueError
import yaml
from ChatBot.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path)->ConfigBox:
    """reads the yaml file and returns

    Args:
        path_to_yaml (Path): path to the yaml file

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox object
    """
    try:
        with open(path_to_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    logger.info(f"yaml file: {path_to_yaml} loaded successfully")
                    return ConfigBox(data)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories
    Args:
        path_to_directories (List): list of directories
        ignore_log (bool, optional): ignore if multiple dirs to be created. Default is False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory {path} created successfully")

@ensure_annotations
def get_size(path:Path)-> str:
    """
    get size in KB
    Args:
        path (Path): path to the file
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"{size_in_kb} KB"