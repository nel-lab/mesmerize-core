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


def make_chunk_projection(Yr_chunk: np.ndarray, proj_type: str):
    return getattr(np, proj_type)(Yr_chunk, axis=1)

def make_chunk_projection_helper(args: tuple[np.ndarray, str]):
    return make_chunk_projection(*args)


def make_projection_parallel(Yr: np.ndarray, dims: tuple[int, ...], T: int,
                             proj_type: str, dview: Optional[Cluster]) -> np.ndarray:
    if dview is None:
        p_img_flat = make_chunk_projection(Yr, proj_type)
    else:
        # use n_pixels_per_process from CNMF to avoid running out of memory
        n_pix = Yr.shape[0]
        p_img_flat = np.empty(n_pix, dtype=Yr.dtype)
        chunk_size = estimate_n_pixels_per_process(get_n_processes(dview), T, dims)
        chunk_starts = range(0, n_pix, chunk_size)
        chunk_slices = [slice(start, min(start + chunk_size, n_pix)) for start in chunk_starts]
        args = ((np.asarray(Yr[chunk_slice]), proj_type) for chunk_slice in chunk_slices)
        for chunk_slice, chunk_proj in zip(chunk_slices, dview.imap(make_chunk_projection_helper, args)):
            p_img_flat[chunk_slice] = chunk_proj
    
    return np.reshape(p_img_flat, dims, order='F')


def save_projections_parallel(uuid, Yr: np.ndarray, dims: tuple[int, ...], T: int,
                              output_dir: Path, dview: Optional[Cluster]) -> dict[str, Path]:
    proj_paths = dict()
    for proj_type in ["mean", "std", "max"]:
        p_img = make_projection_parallel(Yr, dims, T, "nan" + proj_type, dview=dview)
        proj_paths[proj_type] = output_dir.joinpath(
            f"{uuid}_{proj_type}_projection.npy"
        )
        np.save(str(proj_paths[proj_type]), p_img)
    return proj_paths
