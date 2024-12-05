from contextlib import contextmanager
import os
from pathlib import Path
import psutil
from typing import (Optional, Union, Generator, Protocol,
                    Callable, TypeVar, Sequence, Iterable, runtime_checkable)

import caiman as cm
from caiman.cluster import setup_cluster
from ipyparallel import DirectView
from multiprocessing.pool import Pool
import numpy as np
import scipy.stats


RetVal = TypeVar("RetVal")
@runtime_checkable
class CustomCluster(Protocol):
    """
    Protocol for a cluster that is not a multiprocessing pool
    (including ipyparallel.DirectView)
    """

    def map_sync(
        self, fn: Callable[..., RetVal], args: Iterable
    ) -> Sequence[RetVal]: ...

    def __len__(self) -> int:
        """return number of workers"""
        ...


Cluster = Union[Pool,  CustomCluster, DirectView]


def get_n_processes(dview: Optional[Cluster]) -> int:
    """Infer number of processes in a multiprocessing or ipyparallel cluster"""
    if isinstance(dview, Pool):
        assert hasattr(dview, '_processes'), "Pool not keeping track of # of processes?"
        return dview._processes  # type: ignore
    elif dview is not None:
        return len(dview)
    else:
        return 1


@contextmanager
def ensure_server(dview: Optional[Cluster]) -> Generator[tuple[Cluster, int], None, None]:
    """
    Context manager that passes through an existing 'dview' or
    opens up a multiprocessing server if none is passed in.
    If a server was opened, closes it upon exit.
    Usage: `with ensure_server(dview) as (dview, n_processes):`
    """
    if dview is not None:
        yield dview, get_n_processes(dview)
    else:
        # no cluster passed in, so open one
        procs_available = psutil.cpu_count()
        if procs_available is None:
            raise RuntimeError('Cannot determine number of processes')

        if "MESMERIZE_N_PROCESSES" in os.environ.keys():
            try:
                n_processes = int(os.environ["MESMERIZE_N_PROCESSES"])
            except:
                n_processes = procs_available - 1
        else:
            n_processes = procs_available - 1

        # Start cluster for parallel processing
        _, dview, n_processes = setup_cluster(
            backend="multiprocessing", n_processes=n_processes, single_thread=False
        )
        assert isinstance(dview, Pool) and isinstance(n_processes, int), 'setup_cluster with multiprocessing did not return a Pool'
        try:
            yield dview, n_processes
        finally:
            cm.stop_server(dview=dview)


def estimate_n_pixels_per_process(n_processes: int, T: int, dims: tuple[int, ...]) -> int:
    """
    Estimate a safe number of pixels to allocate to each parallel process at a time
    Taken from CNMF.fit (TODO factor this out in caiman and just import it)
    """
    avail_memory_per_process = psutil.virtual_memory()[
        1] / 2.**30 / n_processes
    mem_per_pix = 3.6977678498329843e-09
    npx_per_proc = int(avail_memory_per_process / 8. / mem_per_pix / T)
    npx_per_proc = int(np.minimum(npx_per_proc, np.prod(dims) // n_processes))
    return npx_per_proc


def make_chunk_projection(Yr_chunk: np.ndarray, proj_type: str, ignore_nan=False):
    if hasattr(scipy.stats, proj_type):
        return getattr(scipy.stats, proj_type)(Yr_chunk, axis=1, nan_policy='omit' if ignore_nan else 'propagate')
    
    if hasattr(np, proj_type):
        if ignore_nan:
            if hasattr(np, "nan" + proj_type):
                proj_type = "nan" + proj_type
            else:
                logging.warning(f"NaN-ignoring version of {proj_type} function does not exist; not ignoring NaNs")    
        return getattr(np, proj_type)(Yr_chunk, axis=1)
    
    raise NotImplementedError(f"Projection type '{proj_type}' not implemented")


def make_chunk_projection_helper(args: tuple[Union[str, np.ndarray], slice, str, bool]):
    Yr, chunk_slice, proj_type, ignore_nan = args
    if isinstance(Yr, str):
        Yr, _, _ = cm.load_memmap(Yr)
    return make_chunk_projection(Yr[chunk_slice], proj_type, ignore_nan=ignore_nan)


def make_projection_parallel(movie_or_path: Union[cm.movie, str], proj_type: str, dview: Optional[Cluster],
                             ignore_nan=False) -> np.ndarray:
    if isinstance(movie_or_path, str):
        movie_path = movie_or_path
        Yr, dims, T = cm.load_memmap(movie_path)
    else:
        dview = None  # avoid doing in parallel if we already know it fits into memory
        T = movie_or_path.shape[0]
        dims = movie_or_path.shape[1:]
        Yr = movie_or_path.reshape((T, -1), order='F').T

    if dview is None:
        p_img_flat = make_chunk_projection(Yr, proj_type, ignore_nan=ignore_nan)
    else:
        # use n_pixels_per_process from CNMF to avoid running out of memory
        n_pix = Yr.shape[0]
        chunk_size = estimate_n_pixels_per_process(get_n_processes(dview), T, dims)
        chunk_starts = range(0, n_pix, chunk_size)
        chunk_slices = [slice(start, min(start + chunk_size, n_pix)) for start in chunk_starts]
        args = [(movie_path, chunk_slice, proj_type, ignore_nan) for chunk_slice in chunk_slices]
        map_fn = dview.map if isinstance(dview, Pool) else dview.map_sync
        chunk_projs = map_fn(make_chunk_projection_helper, args)
        p_img_flat = np.concatenate(chunk_projs, axis=0)
    return np.reshape(p_img_flat, dims, order='F')


def save_projections_parallel(uuid, movie_path: Union[str, Path], output_dir: Path, dview: Optional[Cluster]
                              ) -> dict[str, Path]:
    proj_paths = dict()
    for proj_type in ["mean", "std", "max"]:
        p_img = make_projection_parallel(str(movie_path), proj_type, dview=dview, ignore_nan=True)
        proj_paths[proj_type] = output_dir.joinpath(
            f"{uuid}_{proj_type}_projection.npy"
        )
        np.save(str(proj_paths[proj_type]), p_img)
    return proj_paths
