import os
from pathlib import Path
from typing import Union

import pandas as pd

from .algorithms import cnmf, mcorr
from .algorithms import cnmfe
from .utils import validate_path

CURRENT_BATCH_PATH: Path = None  # only one batch at a time
PARENT_DATA_PATH: Path = None

ALGO_MODULES = {
    "cnmf": cnmf,
    "mcorr": mcorr,
    "cnmfe": cnmfe,
}

COMPUTE_BACKEND_SUBPROCESS = "subprocess"  #: subprocess backend
COMPUTE_BACKEND_SLURM = "slurm"  #: SLURM backend, not yet implemented

COMPUTE_BACKENDS = [COMPUTE_BACKEND_SUBPROCESS, COMPUTE_BACKEND_SLURM]

DATAFRAME_COLUMNS = ["algo", "name", "input_movie_path", "params", "outputs", "comments", "uuid"]


def set_parent_raw_data_path(path: Union[Path, str]) -> Path:
    """
    Set the global `PARENT_DATA_PATH`

    Parameters
    ----------
    path: Union[Path, str]
        Full parent data path
    """
    global PARENT_DATA_PATH
    path = validate_path(path)
    PARENT_DATA_PATH = Path(path)

    return PARENT_DATA_PATH


def get_parent_raw_data_path() -> Path:
    """
    Get the global `PARENT_DATA_PATH`
    Returns
    -------
    Path
        global `PARENT_DATA_PATH`

    """
    global PARENT_DATA_PATH
    return PARENT_DATA_PATH


class _BasePathExtensions:
    def __init__(self, data: Union[pd.DataFrame, pd.Series]):
        self._data = data

    def set_batch_path(self, path: Union[str, Path]):
        self._data.attrs["batch_path"] = Path(path)

    def get_batch_path(self) -> Path:
        if "batch_path" in self._data.attrs.keys():
            if self._data.attrs["batch_path"] is not None:
                return self._data.attrs["batch_path"]
        else:
            raise ValueError("Batch path is not set")

    def resolve(self, path: Union[str, Path]) -> Path:
        """
        Resolve the full path of ``path`` if possible, first tries batch_dir then raw_data_dir
        Last resort plain resolve.

        Parameters
        ----------
        path: Union[str, Path]
            The relative path to resolve

        Returns
        -------
        Full path with the batch path or raw data path appended

        """
        path = Path(path)
        # check if input movie is within batch dir
        if self.get_batch_path().parent.joinpath(path).exists():
            return self.get_batch_path().parent.joinpath(path)

        # else check if in parent raw data dir
        elif get_parent_raw_data_path().joinpath(path).exists():
            return get_parent_raw_data_path().joinpath(path)

        else:
            raise FileNotFoundError(f"Could not resolve full path of:\n{path}")

    def split(self, path: Union[str, Path]):
        """
        Split a full path into (batch_dir, relative_path) or (raw_data_dir, relative_path)
        Parameters
        ----------
        path: Union[str, Path]
            Full path to split with respect to batch_dir or raw_data_dir

        Returns
        -------
        Tuple[Path, Path]
            (<batch_dir or raw_data_dir>, <relative_path>)

        """
        path = Path(path)
        # check if input movie is within batch dir
        if self.get_batch_path().parent in path.parents:
            return self.get_batch_path().parent, path.relative_to(
                self.get_batch_path().parent
            )

        # else check if in parent raw data dir
        elif get_parent_raw_data_path() in path.parents:
            return get_parent_raw_data_path(), path.relative_to(
                get_parent_raw_data_path()
            )

        else:
            raise NotADirectoryError(
                f"Could not split `path`:\n{path}"
                f"\nnot relative to either batch path:\n{self.get_batch_path()}"
                f"\nor parent raw data path:\n{get_parent_raw_data_path()}"
            )


@pd.api.extensions.register_dataframe_accessor("paths")
class PathsDataFrameExtension(_BasePathExtensions):
    pass


@pd.api.extensions.register_series_accessor("paths")
class PathsSeriesExtension(_BasePathExtensions):
    pass


def load_batch(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the batch pickle file, also sets the global `CURRENT_BATCH_PATH`

    Parameters
    ----------
    path: Union[str, Path])

    Returns
    -------

    """
    # global CURRENT_BATCH_PATH
    path = validate_path(path)

    df = pd.read_pickle(Path(path))

    # CURRENT_BATCH_PATH = pathlib.Path(batch_file)

    df.paths.set_batch_path(path)

    return df


def create_batch(path: Union[str, Path], remove_existing: bool = False) -> pd.DataFrame:
    """
    Create a new batch DataFrame

    Parameters
    ----------
    path: Union[str. Path]
        path to save the new batch DataFrame

    remove_existing: bool
        If `True`, remove an existing batch DataFrame file if it exists at the given `path`

    Returns
    -------
    pd.DataFrame
        New empty batch DataFrame

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
    df.paths.set_batch_path(path)

    df.to_pickle(path)

    # global CURRENT_BATCH_PATH
    # CURRENT_BATCH_PATH = path

    return df


def get_full_raw_data_path(path: Union[Path, str]) -> Path:
    path = Path(path)
    if PARENT_DATA_PATH is not None:
        return PARENT_DATA_PATH.joinpath(path)

    return path
