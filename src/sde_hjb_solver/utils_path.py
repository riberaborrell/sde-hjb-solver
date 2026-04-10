import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from dotenv import load_dotenv

PathLike = Union[str, os.PathLike]


def get_project_dir() -> Path:
    """Return the absolute path of the repository's directory."""
    return Path(__file__).resolve().parent


def get_data_dir() -> PathLike:
    """Return the absolute path of the repository's data directory."""
    # load .env file
    load_dotenv()
    return os.getenv('SDE_HJB_DATA_DIR', get_project_dir() / 'data')


def make_dir_path(dir_path: PathLike) -> None:
    """Create directories for the given path if they do not already exist.

    Args:
        dir_path: Path to create.
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def empty_dir(dir_path: PathLike) -> None:
    """Remove all files in the directory from the given path.

    Args:
        dir_path: Directory path to empty.
    """
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete {}. Reason: {}'.format((file_path, e)))


def save_data(data_dict: Dict[str, Any], rel_dir_path: PathLike) -> None:
    """Save arrays into a .npz file under the data directory.

    Args:
        data_dict: Mapping of array names to numpy arrays.
        rel_dir_path: Relative directory under the data directory.
    """

    dir_path = os.path.join(get_data_dir(), rel_dir_path)

    # create directoreis of the given path if it does not exist
    make_dir_path(dir_path)

    file_path = os.path.join(dir_path, 'hjb-solution.npz')
    np.savez(file_path, **data_dict)


def load_data(rel_dir_path: PathLike) -> Dict[str, Any]:
    """Load arrays from a .npz file under the data directory.

    Args:
        rel_dir_path: Relative directory under the data directory.

    Returns:
        Mapping of array names to numpy arrays or scalars.
    """
    try:
        file_path = os.path.join(get_data_dir(), rel_dir_path, 'hjb-solution.npz')
        data = dict(np.load(file_path, allow_pickle=True))
        for file_name in data.keys():
            if data[file_name].ndim == 0:
                data[file_name] = data[file_name].item()
        return data
    except FileNotFoundError as e:
        print(e)
        sys.exit()
