import os
import pathlib
from pathlib import Path
from typing import Union

import pandas as pd

from .algorithms import cnmf, mcorr
from .algorithms import cnmfe
from .utils import validate_path

CURRENT_BATCH_PATH: pathlib.Path = None  # only one batch at a time
PARENT_DATA_PATH: pathlib.Path = None

ALGO_MODULES = \
    {
        'cnmf': cnmf,
        'mcorr': mcorr,
        'cnmfe': cnmfe,
    }

COMPUTE_BACKEND_QPROCESS = 'qprocess'  #: QProcess backend for use in napari
COMPUTE_BACKEND_SUBPROCESS = 'subprocess'  #: subprocess backend, for use output a Qt application such as a notebook
COMPUTE_BACKEND_SLURM = 'slurm'  #: SLURM backend, not yet implemented

COMPUTE_BACKENDS =\
[
    COMPUTE_BACKEND_QPROCESS,
    COMPUTE_BACKEND_SUBPROCESS,
    COMPUTE_BACKEND_SLURM
]

DATAFRAME_COLUMNS = ['algo', 'name', 'input_movie_path', 'params', 'outputs', 'uuid']


class BasePaths:
    def __init__(self):
        self._batch_path = None
        self._parent_path = None

    def set_batch_path(self, path: Union[str, Path]):
        self._batch_path = Path(path)

    def get_batch_path(self) -> Path:
        if self._batch_path is not None:
            return self._batch_path
        else:
            raise ValueError('Batch path is not set')

    def set_parent_path(self, path: Union[str, Path]):
        self._parent_path = path

    def get_parent_path(self) -> Path:
        if self._parent_path is not None:
            return self._parent_path
        else:
            raise ValueError('Parent path is not set')


@pd.api.extensions.register_dataframe_accessor("paths")
class PathsDataFrameExtension(BasePaths):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        BasePaths.__init__(self)


def set_parent_data_path(path: Union[Path, str]) -> Path:
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


def get_parent_data_path() -> Path:
    """
    Get the global `PARENT_DATA_PATH`
    Returns
    -------
    Path
        global `PARENT_DATA_PATH`

    """
    global PARENT_DATA_PATH
    return PARENT_DATA_PATH


def load_batch(batch_file: Union[str, pathlib.Path]) -> pd.DataFrame:
    """
    Load the batch pickle file, also sets the global `CURRENT_BATCH_PATH`

    Parameters
    ----------
    batch_file: Union[str, Path])

    Returns
    -------

    """
    # global CURRENT_BATCH_PATH
    batch_file = validate_path(batch_file)

    df = pd.read_pickle(
        pathlib.Path(batch_file)
    )

    # CURRENT_BATCH_PATH = pathlib.Path(batch_file)

    df.caiman.path = batch_file
    df.paths.set_batch_path(batch_file)

    return df


def create_batch(path: str = None, remove_existing: bool = False) -> pd.DataFrame:
    """
    Create a new batch DataFrame

    Parameters
    ----------
    path: str
        path to save the new batch DataFrame

    remove_existing: bool
        If `True`, remove an existing batch DataFrame file if it exists at the given `path`

    Returns
    -------
    pd.DataFrame
        New empty batch DataFrame

    """
    path = validate_path(path)

    if pathlib.Path(path).is_file():
        if remove_existing:
            os.remove(path)
        else:
            raise FileExistsError(
                f'Batch file already exists at specified location: {path}'
            )

    if not Path(path).parent.is_dir():
        os.makedirs(Path(path).parent)

    df = pd.DataFrame(columns=DATAFRAME_COLUMNS)
    df.caiman.path = path
    df.paths.set_batch_path(path)

    df.to_pickle(path)

    # global CURRENT_BATCH_PATH
    # CURRENT_BATCH_PATH = path

    return df


def get_full_data_path(path: Union[Path, str]) -> Path:
    path = Path(path)
    if PARENT_DATA_PATH is not None:
        return PARENT_DATA_PATH.joinpath(path)

    return path


