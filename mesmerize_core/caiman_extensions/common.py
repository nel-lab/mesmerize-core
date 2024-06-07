import os
import shutil
from pathlib import Path
import psutil
from subprocess import Popen
from typing import *
from uuid import UUID, uuid4
from shutil import rmtree
from itertools import chain
from collections import Counter
from datetime import datetime
import time
from copy import deepcopy

import numpy as np
import pandas as pd

from ._batch_exceptions import BatchItemNotRunError, BatchItemUnsuccessfulError, DependencyError
from ._utils import validate, _index_parser
from ..batch_utils import (
    COMPUTE_BACKENDS,
    COMPUTE_BACKEND_SUBPROCESS,
    COMPUTE_BACKEND_LOCAL,
    get_parent_raw_data_path,
    load_batch
)
from ..utils import validate_path, IS_WINDOWS, make_runfile, warning_experimental
from .cnmf import cnmf_cache
from .. import algorithms
from ..movie_readers import default_reader


ALGO_MODULES = {
    "cnmf": algorithms.cnmf,
    "mcorr": algorithms.mcorr,
    "cnmfe": algorithms.cnmfe,
}


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

    def add_item(self, algo: str, item_name: str, input_movie_path: Union[str, pd.Series], params: dict):
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

        if isinstance(input_movie_path, pd.Series):
            if not input_movie_path["algo"] == "mcorr":
                raise ValueError(
                    "`input_movie_path` argument must be an input movie path "
                    "as a `str` or `Path` object, or a mcorr batch item."
                )
            input_movie_path = input_movie_path.mcorr.get_output_path()

        # make sure path is within batch dir or parent raw data path
        input_movie_path = self._df.paths.resolve(input_movie_path)
        validate_path(input_movie_path)

        # get relative path
        input_movie_path = self._df.paths.split(input_movie_path)[1]

        # convert lists to tuples so that get_params_diffs works
        for k in list(params["main"].keys()):
            if isinstance(params["main"][k], list):
                params["main"][k] = tuple(params["main"][k])

        # Create a pandas Series (Row) with the provided arguments
        s = pd.Series(
            {
                "algo": algo,
                "item_name": item_name,
                "input_movie_path": str(input_movie_path),
                "params": params,
                "outputs": None,  # to store dict of output information, such as output file paths
                "added_time": datetime.now().isoformat(timespec="seconds", sep="T"),
                "ran_time": None,
                "algo_duration": None,
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

    def save_to_disk(self, max_index_diff: int = 0):
        """
        Saves DataFrame to disk, copies to a backup before overwriting existing file.
        """
        path: Path = self._df.paths.get_batch_path()

        disk_df = load_batch(path)

        # check that max_index_diff is not exceeded
        if abs(disk_df.index.size - self._df.index.size) > max_index_diff:
            raise IndexError(
                f"The number of rows in the DataFrame on disk differs more "
                f"than has been allowed by the `max_index_diff` kwarg which "
                f"is set to <{max_index_diff}>. This is to prevent overwriting "
                f"the full DataFrame with a sub-DataFrame. If you still wish "
                f"to save the smaller DataFrame, use `caiman.save_to_disk()` "
                f"with `max_index_diff` set to the highest allowable difference "
                f"in row number."
            )

        bak = path.with_suffix(path.suffix + f"bak.{time.time()}")

        shutil.copyfile(path, bak)
        try:
            self._df.to_pickle(path)
            os.remove(bak)
        except:
            shutil.copyfile(bak, path)
            raise IOError(f"Could not save dataframe to disk.")

    def reload_from_disk(self) -> pd.DataFrame:
        """
        Returns the DataFrame on disk.

        Example:

            .. code-block:: python

                df = df.caiman.reload_from_disk()

        Returns
        -------
        pd.DataFrame

        """
        return load_batch(self._df.paths.get_batch_path())

    @_index_parser
    def remove_item(self, index: Union[int, str, UUID], remove_data: bool = True, safe_removal: bool = True):
        """
        Remove a batch item from the DataFrame and delete all data associated
        to that batch item from disk if ``remove_data=True``

        Parameters
        ----------
        index: int, str or UUID
            The index of the batch item to remove from the DataFrame as a numerical ``int`` index, ``str`` representing
            a UUID, or a UUID object.

        remove_data: bool
            if ``True`` removes all output data associated to the batch item from disk.
            The input movie located at ``input_movie_path`` is not affected.

        safe_removal: bool
            if ``True``, this batch item is not removed and raises an exception if the output of this batch
            item is the input to another batch item. For example, if this is a *motion correction* batch item whose
            output is used as the input movie for a *CNMF* batch item.

            | if ``False``, this batch item is removed even if its output is the input to another batch item

        Returns
        -------

        """
        if self._df.iloc[index]["outputs"] is not None:
            if self._df.iloc[index]["outputs"]["success"] is True:
                # if success, don't skip checks
                skip_checks = False
            else:
                # if failed item, skip checks, allow removal since it cannot have children
                skip_checks = True
        else:
            # if not run, skip checks, it cannot have children
            skip_checks = True

        if self._df.iloc[index]["algo"] == "mcorr" and not skip_checks:
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
            except FileNotFoundError:
                pass

        # Drop selected index
        self._df.drop([index], inplace=True)
        # Reset indices so there are no 'jumps'
        self._df.reset_index(drop=True, inplace=True)
        # Save new df to disc
        self._df.to_pickle(self._df.paths.get_batch_path())

    def get_params_diffs(self, algo: str, item_name: str) -> pd.DataFrame:
        """
        Get the parameters that differ for a given `item_name` run with a given `algo`

        Parameters
        ----------
        algo: str
            algorithm, one of "mcorr", "cnmf", or "cnmfe"

        item_name: str
            The item name for which to get the parameter diffs

        Returns
        -------
        pd.DataFrame
            pandas DataFrame) with dicts containing only the
            parameters that vary between batch items for the given
            `item_name`. The returned index corresponds to the
            index of the original DataFrame

        """            
        def flatten_params(params_dict: dict):
            """
            Produce a flat dict with one entry for each parameter in the passed dict.
            If params_dict['main'] is nested one level (e.g., {'init': {'K': 5}, 'merging': {'merge_thr': 0.85}}...),
            each key in the output is <outerKey>.<innerKey>, e.g., [(init.K, 5), (merging.merge_thr, 0.85)]
            """
            params = {}
            for key1, val1 in params_dict.items():
                if isinstance(val1, dict):  # nested
                    for key2, val2 in val1.items():
                        params[f"{key1}.{key2}"] = val2
                else:
                    params[key1] = val1
            return params
        
        sub_df = self._df[self._df["item_name"] == item_name]
        sub_df = sub_df[sub_df["algo"] == algo]

        if sub_df.index.size == 0:
            raise NameError(f"The given `item_name`: {item_name}, does not exist in the DataFrame")

        # get flattened parameters for each of the filtered items
        params_flat = sub_df.params.map(lambda p: flatten_params(p["main"]))

        # build list of params that differ between different parameter sets
        common_params = deepcopy(params_flat.iat[0])  # holds the common value for parameters found in all sets (so far)
        varying_params = set()  # set of parameter keys that appear in not all sets or with varying values

        for this_params in params_flat.iloc[1:]:
            # first, anything that's not in both this dict and the common set is considered varying
            common_paramset = set(common_params.keys())
            for not_common_key in common_paramset.symmetric_difference(this_params.keys()):
                varying_params.add(not_common_key)
                if not_common_key in common_paramset:
                    del common_params[not_common_key]
                    common_paramset.remove(not_common_key)

            # second, look at params in the common set and remove any that differ for this set
            for key in common_paramset:  # iterate over this set rather than dict itself to avoid issues when deleting entries
                if not np.array_equal(common_params[key], this_params[key]):  # (should also work for scalars/arbitrary objects)
                    varying_params.add(key)
                    del common_params[key]

        # gives a list where each item is a dict that has the unique params that correspond to a row
        # the indices of this series correspond to the index of the row in the parent dataframe
        diffs = params_flat.map(lambda p: {key: p[key] for key in varying_params if key in p})

        # return as a nicely formatted dataframe
        diffs_df = pd.DataFrame.from_dict(diffs.tolist(), dtype=object).set_index(diffs.index)

        # replace any missing parameters with a string for clarity
        with pd.option_context('future.no_silent_downcasting', True):  # avoids warning about downcasting
            diffs_df.fillna("<default>", inplace=True)

        return diffs_df

    @warning_experimental("This feature will change in the future and directly return the "
                          " a DataFrame of children (rows, ie. child batch items row) "
                          "instead of a list of UUIDs")
    @_index_parser
    def get_children(self, index: Union[int, str, UUID]) -> List[UUID]:
        """
        For the *motion correction* batch item at the provided ``index``,
        returns a list of UUIDs for *CNMF(E)* batch items that use the
        output of this motion correction batch item.

        | Note: Only Motion Correction items have children, CNMF(E) items do not have children.

        Parameters
        ----------
        index: int, str, or UUID
            the index of the mcorr item to get the children of, provided as a numerical ``int`` index, str representing
            a UUID, or a UUID object

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

    @warning_experimental("This feature will change in the future and directly return the "
                          " pandas.Series (row, ie. batch item row) instead of the UUID")
    @_index_parser
    def get_parent(self, index: Union[int, str, UUID]) -> Union[UUID, None]:
        """
        Get the UUID of the batch item whose output was used as
        the input for the batch item at the provided ``index``.

        | If a parent exists, it is always an mcorr batch item

        Parameters
        ----------
        index: int, str, or UUID
            the index of the batch item to get the parent of, provided as a numerical ``int`` index, str representing
            a UUID, or a UUID object

        Returns
        -------
        UUID or None
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


class DummyProcess:
    """Dummy process for local backend"""
    def wait(self):
        pass


@pd.api.extensions.register_series_accessor("caiman")
class CaimanSeriesExtensions:
    """
    Extensions for caiman stuff
    """

    def __init__(self, s: pd.Series):
        self._series = s
        self.process: Popen = None

    def _run_local(
            self,
            algo: str,
            batch_path: Path,
            uuid: UUID,
            data_path: Union[Path, None],
    ):
        algo_module = getattr(algorithms, algo)
        algo_module.run_algo(
            batch_path=str(batch_path),
            uuid=str(uuid),
            data_path=str(data_path)
        )

        return DummyProcess()

    def _run_subprocess(
        self,
        runfile_path: str,
        wait: bool,
        **kwargs
    ):

        # Get the dir that contains the input movie
        parent_path = self._series.paths.resolve(self._series.input_movie_path).parent
        if not IS_WINDOWS:
            self.process = Popen(runfile_path, cwd=parent_path)
        else:
            self.process = Popen(f"powershell {runfile_path}", cwd=parent_path)

        if wait:
            self.process.wait()

        return self.process

    def _run_slurm(
        self,
        runfile_path: str,
        wait: bool,
        partition: Optional[Union[str, list[str]]] = None,
        **kwargs
    ):
        """
        Run on a cluster using SLURM. Configurable options (to pass to run):
        - partition: if given, tells SLRUM to run the job on the given partition(s).
        """

        # this needs to match what's in the runfile
        if 'MESMERIZE_N_PROCESSES' in os.environ:
            n_procs = os.environ['MESMERIZE_N_PROCESSES']
        else:
            n_procs = psutil.cpu_count() - 1

        # make sure we have a place to save log files
        uuid = str(self._series["uuid"])
        output_dir = Path(runfile_path).parent.joinpath(uuid)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{uuid}.log'

        # --wait means that the lifetme of the created process corresponds to the lifetime of the job
        submission_opts = (f'--job-name={self._series["algo"]}-{uuid[:8]} --ntasks=1 ' +
            f'--cpus-per-task={n_procs} --output={output_path} --wait')
        
        if partition is not None:
            if isinstance(partition, str):
                partition = [partition]
            submission_opts += f' --partition={",".join(partition)}'

        self.process = Popen(['sbatch', *submission_opts.split(" "), runfile_path])
        if wait:
            self.process.wait()
        
        return self.process

    @cnmf_cache.invalidate()
    def run(
            self,
            backend: Optional[str] = None,
            wait: bool = True,
            **kwargs
    ):
        """
        Run a CaImAn algorithm in an external process using the chosen backend

        NoRMCorre, CNMF, or CNMFE will be run for this Series.
        Each Series (DataFrame row) has a `input_movie_path` and `params` for the algorithm

        Parameters
        ----------
        backend: str, optional
            One of the available backends, default on Linux & Mac is ``"subprocess"``. Default on Windows is
            ``"local"`` since Windows is inconsistent in the way it launches subprocesses

        wait: bool, default ``True``
            if using the ``"subprocess"`` backend, call ``wait()`` on the ``Popen`` instance before returning it

        **kwargs
            any kwargs to pass to the backend
        """
        if get_parent_raw_data_path() is None:
            raise ValueError(
                "parent raw data path is not set, you must set it using:\n"
                "`set_parent_raw_data_path()`"
            )

        if backend is None:
            if not IS_WINDOWS:
                backend = COMPUTE_BACKEND_SUBPROCESS
            else:
                backend = COMPUTE_BACKEND_LOCAL

        if backend not in COMPUTE_BACKENDS:
            raise KeyError(
                f"Invalid or unavailable `backend`, choose from the following backends:\n"
                f"{COMPUTE_BACKENDS}"
            )

        batch_path = self._series.paths.get_batch_path()

        if backend == COMPUTE_BACKEND_LOCAL:
            print(f"Running {self._series.uuid} with local backend")
            return self._run_local(
                algo=self._series["algo"],
                batch_path=batch_path,
                uuid=self._series["uuid"],
                data_path=get_parent_raw_data_path(),
            )

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
                runfile_path, wait=wait, **kwargs
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

    def get_input_movie(self, reader: callable = None, **kwargs) -> Union[np.ndarray, Any]:
        """
        Get the input movie

        Parameters
        ----------
        reader: callable
            a function that take the input movie path and return an array-like

        **kwargs
            passed to ``reader`` function

        Returns
        -------

        """
        path_str = str(self.get_input_movie_path())

        if reader is not None:
            if not callable(reader):
                raise TypeError(
                    f"reader must be a callable type, such as a function"
                )

            return reader(path_str, **kwargs)

        return default_reader(path_str, **kwargs)

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
        """
        Return the ``max``, ``mean``, or ``std`` (standard deviation) projection
        
        Parameters
        ----------
        proj_type: str
            one of ``"max"``, ``"mean"``, ``"std"``

        Returns
        -------
        np.ndarray
            ``max``, ``mean``, or ``std`` projection
        """
        path = self._series.paths.resolve(
            self._series["outputs"][f"{proj_type}-projection-path"]
        )
        return np.load(path)
