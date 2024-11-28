from datetime import datetime
import os
from pathlib import Path
from typing import Union

import pandas as pd

from .utils import validate_path

CURRENT_BATCH_PATH: Path = None  # only one batch at a time
PARENT_DATA_PATH: Path = None

COMPUTE_BACKEND_SUBPROCESS = "subprocess"  #: subprocess backend
COMPUTE_BACKEND_SLURM = "slurm"  #: SLURM backend
COMPUTE_BACKEND_LOCAL = "local"

COMPUTE_BACKENDS = [COMPUTE_BACKEND_SUBPROCESS, COMPUTE_BACKEND_SLURM, COMPUTE_BACKEND_LOCAL]

DATAFRAME_COLUMNS = ["algo", "item_name", "input_movie_path", "params", "outputs", "added_time", "ran_time", "algo_duration", "comments", "uuid"]


def set_parent_raw_data_path(path: Union[Path, str]) -> Path:
    """
    Set the global `PARENT_DATA_PATH`

    Parameters
    ----------
    path: Path or str
        Full parent data path
    """
    global PARENT_DATA_PATH
    path = Path(validate_path(path))
    if not path.is_dir():
        raise NotADirectoryError(
            "The directory passed to `set_parent_raw_data_path()` does not exist.\n"
        )
    PARENT_DATA_PATH = path

    return PARENT_DATA_PATH


def get_parent_raw_data_path() -> Path:
    """
    Get the global `PARENT_DATA_PATH`

    Returns
    -------
    Path
        global `PARENT_DATA_PATH` as a Path object

    """
    global PARENT_DATA_PATH
    return PARENT_DATA_PATH


class _BasePathExtensions:
    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        self._data = data

    def set_batch_path(self, path: Union[str, Path]):
        self._data.attrs["batch_path"] = Path(path)

    def get_batch_path(self) -> Path:
        """
        Get the full path to the batch dataframe file

        Returns
        -------
        Path
            full path to the batch dataframe file as a Path object
        """
        if "batch_path" in self._data.attrs.keys():
            if self._data.attrs["batch_path"] is not None:
                return self._data.attrs["batch_path"]
        else:
            raise ValueError("Batch path is not set")

    def resolve(self, path: Union[str, Path]) -> Path:
        """
        Resolve the full path of the passed ``path`` if possible, first tries
        "batch_dir" then "raw_data_dir".

        Parameters
        ----------
        path: str or Path
            The relative path to resolve

        Returns
        -------
        Path
            Full path with the batch path or raw data path appended

        """
        path = Path(path)
        # check if input movie is within batch dir
        if self.get_batch_path().parent.joinpath(path).exists():
            return self.get_batch_path().parent.joinpath(path)

        # else check if in parent raw data dir
        elif get_parent_raw_data_path() is not None:
            if get_parent_raw_data_path().joinpath(path).exists():
                return get_parent_raw_data_path().joinpath(path)

        raise FileNotFoundError(f"Could not resolve full path of:\n{path}")

    def split(self, path: Union[str, Path]):
        """
        Split a full path into (batch_dir, relative_path) or (raw_data_dir, relative_path)

        Parameters
        ----------
        path: str or Path
            Full path to split with respect to batch_dir or raw_data_dir

        Returns
        -------
        Tuple[Path, Path]
            (<batch_dir> or <raw_data_dir>, <relative_path>)

        """
        path = Path(path).resolve()
        # check if input movie is within batch dir
        batch_parent_abs = self.get_batch_path().parent.resolve()
        if batch_parent_abs in path.parents:
            return batch_parent_abs, path.relative_to(batch_parent_abs)

        # else check if in parent raw data dir
        else:
            raw_data_path = get_parent_raw_data_path()
            if raw_data_path is not None:
                raw_data_path_abs = raw_data_path.resolve()
                if raw_data_path_abs in path.parents:
                    return raw_data_path_abs, path.relative_to(raw_data_path_abs)

        raise NotADirectoryError(
                f"Could not split `path`:\n{path}"
                f"\nnot relative to either batch path:\n{batch_parent_abs}"
                f"\nor parent raw data path:\n{raw_data_path_abs if raw_data_path is not None else '(not set)'}"
            )


@pd.api.extensions.register_dataframe_accessor("paths")
class PathsDataFrameExtension(_BasePathExtensions):
    pass


@pd.api.extensions.register_series_accessor("paths")
class PathsSeriesExtension(_BasePathExtensions):
    pass


def load_batch(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the batch dataframe pickle file

    Parameters
    ----------
    path: str or Path

    Returns
    -------
    pd.DataFrame
        batch dataframe loaded from the specified path

    Examples
    --------

    .. code-block:: python

        from mesmerize_core import *

        df = load_batch("/path/to/batch.pickle")

        # view dataframe
        df.head()

    """

    path = validate_path(path)

    df = pd.read_pickle(Path(path))

    df.paths.set_batch_path(path)

    # check to see if added and ran timestamp are in df
    if all(item in df.columns for item in ["added_time", "ran_time", "algo_duration"]):
        return df
    else:
        df["added_time"] = None
        df["ran_time"] = None
        df["algo_duration"] = None
        return df


def create_batch(path: Union[str, Path], remove_existing: bool = False) -> pd.DataFrame:
    """
    Create a new batch DataFrame

    Parameters
    ----------
    path: str or Path
        path to save the new batch DataFrame as a pickle file

    remove_existing: bool
        If ``True``, remove an existing batch DataFrame file if it exists at the given `path`, default ``False``

    Returns
    -------
    pd.DataFrame
        New empty batch DataFrame

    Examples
    --------

    .. code-block:: python

        from mesmerize_core import *

        df = create_batch("/path/to/new_batch.pickle")

    """
    path = validate_path(path)

    if Path(path).is_file():
        if remove_existing:
            os.remove(path)
        else:
            raise FileExistsError(
                f"Batch file already exists at specified location: {path}"
            )

    if not Path(path).parent.is_dir():
        os.makedirs(Path(path).parent)

    df = pd.DataFrame(columns=DATAFRAME_COLUMNS)
    df.to_pickle(path)  # save before adding platform-dependent batch path
     
    df.paths.set_batch_path(path)

    return df


def get_full_raw_data_path(path: Union[Path, str]) -> Path:
    path = Path(path)
    if PARENT_DATA_PATH is not None:
        return PARENT_DATA_PATH.joinpath(path)

    return path
