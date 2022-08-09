import os
from functools import partial, wraps
from pathlib import Path
from subprocess import Popen
from typing import Union, List, Optional
from uuid import UUID, uuid4
from shutil import rmtree

import numpy as np
import pandas as pd
import pims

from ..batch_utils import (
    COMPUTE_BACKENDS,
    COMPUTE_BACKEND_SUBPROCESS,
    ALGO_MODULES,
    get_parent_raw_data_path,
    PathsDataFrameExtension,
    PathsSeriesExtension,
)
from ..utils import validate_path, IS_WINDOWS, make_runfile, warning_experimental
from caiman import load_memmap


class BatchItemNotRunError(Exception):
    pass


class BatchItemUnsuccessfulError(Exception):
    pass


class WrongAlgorithmExtensionError(Exception):
    pass


class DependencyError(Exception):
    pass


def validate(algo: str = None):
    def dec(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._series["outputs"] is None:
                raise BatchItemNotRunError("Item has not been run")

            if algo is not None:
                if algo not in self._series["algo"]:
                    raise WrongAlgorithmExtensionError(
                        f"<{algo} extension called for a <{self._series}> item"
                    )

            if not self._series["outputs"]["success"]:
                tb = self._series["outputs"]["traceback"]
                raise BatchItemUnsuccessfulError(f"Batch item was unsuccessful, traceback from subprocess:\n{tb}")
            return func(self, *args, **kwargs)

        return wrapper

    return dec


def _index_parser(func):
    @wraps(func)
    def _parser(instance, *args, **kwargs):
        if "index" in kwargs.keys():
            index: Union[int, str, UUID] = kwargs["index"]
        elif len(args) > 0:
            index = args[0]  # always first positional arg

        if isinstance(index, (UUID, str)):
            _index = instance._df[instance._df["uuid"] == str(index)].index
            if _index.size == 0:
                raise ValueError(f"No batch item found with uuid: {index}")

            index = _index.item()

        if not isinstance(index, int):
            raise TypeError(f"`index` argument must be of type `int`, `str`, or `UUID`")

        if "index" in kwargs.keys():
            kwargs["index"] = index
        else:
            args = (index, *args[1:])

        return func(instance, *args, **kwargs)
    return _parser


@pd.api.extensions.register_dataframe_accessor("caiman")
class CaimanDataFrameExtensions:
    """
    Extensions for caiman related functions
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def uloc(self, u: Union[str, UUID]) -> pd.Series:
        """
        Return the series corresponding to the passed UUID
        """
        df_u = self._df.loc[self._df["uuid"] == str(u)]

        if df_u.index.size == 0:
            raise KeyError("Item with given UUID not found in dataframe")
        elif df_u.index.size > 1:
            raise KeyError(
                f"Duplicate items with given UUID found in dataframe, something is wrong\n"
                f"{df_u}"
            )

        return df_u.squeeze()

    def add_item(self, algo: str, item_name: str, input_movie_path: str, params: dict):
        """
        Add an item to the DataFrame to organize parameters
        that can be used to run a CaImAn algorithm

        Parameters
        ----------
        algo: str
            Name of the algorithm to run, one of ``"mcorr"``, ``"cnmf"`` or ``"cnmfe"``

        item_name: str
            User set name for the batch item

        input_movie_path: str
            Full path to the input movie

        params:
            Parameters for running the algorithm with the input movie

        """
        if get_parent_raw_data_path() is None:
            raise ValueError(
                "parent raw data path is not set, you must set it using:\n"
                "`set_parent_raw_data_path()`"
            )

        # make sure path is within batch dir or parent raw data path
        input_movie_path = self._df.paths.resolve(input_movie_path)
        validate_path(input_movie_path)

        # get relative path
        input_movie_path = self._df.paths.split(input_movie_path)[1]

        # Create a pandas Series (Row) with the provided arguments
        s = pd.Series(
            {
                "algo": algo,
                "item_name": item_name,
                "input_movie_path": str(input_movie_path),
                "params": params,
                "outputs": None,  # to store dict of output information, such as output file paths
                "comments": None,
                "uuid": str(
                    uuid4()
                ),  # unique identifier for this combination of movie + params
            }
        )

        # Add the Series to the DataFrame
        self._df.loc[self._df.index.size] = s

        # Save DataFrame to disk
        self._df.to_pickle(self._df.paths.get_batch_path())

    @_index_parser
    def remove_item(self, index: Union[int, str, UUID], remove_data: bool = True, safe_removal: bool = True):
        """
        Remove a batch item from the DataFrame and delete all data associated
        to that batch item from disk if ``remove_data=True``

        Parameters
        ----------
        index: Union[int, str, UUID]
            The index of the batch item to remove from the DataFrame

        remove_data: bool
            If ``True`` removes all output data associated to the batch item from disk.
            | the input movie located at ``input_movie_path`` is not affect.

        safe_removal: bool
            | if ``True``, this batch item is not removed and raises an exception if the output of this batch
            item is the input to another batch item. For example, if this is a *motion correction* batch item whose
            output is used as the input movie for a *CNMF* batch item.
            | if ``False``, this batch item is removed even if its output is the input to another batch item

        Returns
        -------

        """
        
        if safe_removal:
            children = self.get_children(index)
            if len(children) > 0:
                raise DependencyError(
                    f"This batch item's output is used as the input for batch items with the following UUIDs:\n"
                    f"{children}\n"
                    f"If you still want to force removal of this batch item use `safe_removal=False`"
                )

        u = self._df.iloc[index]["uuid"]

        if remove_data:
            try:
                rmtree(self._df.paths.get_batch_path().parent.joinpath(u))
            except PermissionError:
                raise PermissionError(
                    "You do not have permissions to remove the "
                    "output data for the batch item, aborting."
                )

        # Drop selected index
        self._df.drop([index], inplace=True)
        # Reset indices so there are no 'jumps'
        self._df.reset_index(drop=True, inplace=True)
        # Save new df to disc
        self._df.to_pickle(self._df.paths.get_batch_path())

    @_index_parser
    def get_children(self, index: Union[int, str, UUID]) -> List[UUID]:
        """
        For the *motion correction* batch item at the provided ``index``,
        returns a list of UUIDs for *CNMF(E)* batch items that use the
        output of this motion correction batch item.

        | Note: Only Motion Correction items have children, CNMF(E) items do not have children.

        Parameters
        ----------
        index: Union[int, str, UUID]
            the index of the mcorr item to get the children of

        Returns
        -------
        List[UUID]
            List of UUIDs of child CNMF items

        """

        if not self._df.iloc[index]["algo"] == "mcorr":
            raise TypeError(
                "`caiman.get_children()` extension maybe only be used with "
                "mcorr batch items, CNMF(E) items do not have children."
            )

        # get the output path for this mcorr item
        output_path = self._df.iloc[index].mcorr.get_output_path()

        # see if this output path shows up in the input_movie_path of any other batch item
        children = list()
        for i, r in self._df.iterrows():
            try:
                _potential_child = r.caiman.get_input_movie_path()
            except FileNotFoundError:  # assume it is not a child anyways
                continue
            if _potential_child == output_path:
                children.append(r["uuid"])
        return children

    @_index_parser
    def get_parent(self, index: Union[int, str, UUID]) -> Union[UUID, None]:
        """
        Get the UUID of the batch item whose output was used as
        the input for the batch item at the provided ``index``.

        | If a parent exists, it is always an mcorr batch item

        Parameters
        ----------
        index: Union[int, str, UUID]
            the index of the batch item to get the parent of

        Returns
        -------
        Union[UUID, None]
            | if ``UUID``, this is the UUID of the batch item whose output was used for the input of the batch item at
            the provided ``index``
            | if ``None``, the batch item at the provided ``index`` has no parent within the batch dataframe.

        """
        input_movie_path = self._df.iloc[index].caiman.get_input_movie_path()

        for i, r in self._df.iterrows():
            if not r["algo"] == "mcorr":
                continue
            try:
                _potential_parent = r.mcorr.get_output_path()
            except (FileNotFoundError, BatchItemUnsuccessfulError, BatchItemNotRunError):
                continue  # can't be a parent if it was unsuccessful

            if _potential_parent == input_movie_path:
                return r["uuid"]


@pd.api.extensions.register_series_accessor("caiman")
class CaimanSeriesExtensions:
    """
    Extensions for caiman stuff
    """

    def __init__(self, s: pd.Series):
        self._series = s
        self.process: Popen = None

    def _run_subprocess(
        self,
        runfile_path: str,
        **kwargs
    ):

        # Get the dir that contains the input movie
        parent_path = self._series.paths.resolve(self._series.input_movie_path).parent
        if not IS_WINDOWS:
            self.process = Popen(runfile_path, cwd=parent_path)
        else:
            self.process = Popen(f"powershell {runfile_path}", cwd=parent_path)
        return self.process

    def _run_slurm(
        self,
        runfile_path: str,
        **kwargs
    ):
        raise NotImplementedError("Not just implemented, just a placeholder")
        # submission_command = (
        #     f'sbatch --ntasks=1 --cpus-per-task=16 --mem=90000 --wrap="{runfile_path}"'
        # )
        #
        # Popen(submission_command.split(" "))

    def run(
        self,
        backend: Optional[str] = COMPUTE_BACKEND_SUBPROCESS, **kwargs
    ):
        """
        Run a CaImAn algorithm in an external process using the chosen backend

        NoRMCorre, CNMF, or CNMFE will be run for this Series.
        Each Series (DataFrame row) has a `input_movie_path` and `params` for the algorithm

        Parameters
        ----------
        backend: Optional[str]
            One of the available backends, if none default is `COMPUTE_BACKEND_SUBPROCESS`

        **kwargs
            any kwargs to pass to the backend
        """
        if get_parent_raw_data_path() is None:
            raise ValueError(
                "parent raw data path is not set, you must set it using:\n"
                "`set_parent_raw_data_path()`"
            )

        if backend not in COMPUTE_BACKENDS:
            raise KeyError(
                f"Invalid or unavailable `backend`, choose from the following backends:\n"
                f"{COMPUTE_BACKENDS}"
            )

        batch_path = self._series.paths.get_batch_path()

        # Create the runfile in the batch dir using this Series' UUID as the filename
        if IS_WINDOWS:
            runfile_ext = ".ps1"
        else:
            runfile_ext = ".runfile"
        runfile_path = str(
            batch_path.parent.joinpath(self._series["uuid"] + runfile_ext)
        )

        args_str = f"--batch-path {batch_path} --uuid {self._series.uuid}"
        if get_parent_raw_data_path() is not None:
            args_str += f" --data-path {get_parent_raw_data_path()}"

        # make the runfile
        runfile_path = make_runfile(
            module_path=os.path.abspath(
                ALGO_MODULES[self._series["algo"]].__file__
            ),  # caiman algorithm
            filename=runfile_path,  # path to create runfile
            args_str=args_str,
        )
        try:
            self.process = getattr(self, f"_run_{backend}")(
                runfile_path, **kwargs
            )
        except:
            with open(runfile_path, "r") as f:
                raise ValueError(f.read())

        return self.process

    def get_input_movie_path(self) -> Path:
        """
        Returns
        -------
        Path
            full path to the input movie file
        """

        return self._series.paths.resolve(self._series["input_movie_path"])

    @warning_experimental()
    def get_input_movie(self) -> Union[np.ndarray, pims.FramesSequence]:
        extension = self.get_input_movie_path().suffixes[-1]

        if extension in [".tiff", ".tif", ".btf"]:
            return pims.open(str(self.get_input_movie_path()))

        elif extension in [".mmap", ".memmap"]:
            Yr, dims, T = load_memmap(str(self.get_input_movie_path()))
            return np.reshape(Yr.T, [T] + list(dims), order="F")

    @validate()
    def get_corr_image(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            correlation image
        """
        path = self._series.paths.resolve(self._series["outputs"]["corr-img-path"])
        return np.load(str(path))

    @validate()
    def get_pnr_image(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            pnr image
        """
        path = self._series.paths.resolve(self._series["outputs"]["pnr-image-path"])
        return np.load(str(path))

    @validate()
    def get_projection(self, proj_type: str) -> np.ndarray:
        path = self._series.paths.resolve(
            self._series["outputs"][f"{proj_type}-projection-path"]
        )
        return np.load(path)
