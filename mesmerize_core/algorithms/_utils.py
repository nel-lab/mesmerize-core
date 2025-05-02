from contextlib import contextmanager
import logging
import math
import os
from pathlib import Path
import psutil
from typing import (Optional, Union, Generator, Protocol,
                    Callable, TypeVar, Sequence, Iterable, runtime_checkable)

import caiman as cm
from caiman.base.movies import get_file_size
from caiman.cluster import setup_cluster
from caiman.summary_images import local_correlations
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


def avail_bytes_per_process(n_processes: int):
    return psutil.virtual_memory()[1] / n_processes


def estimate_n_pixels_per_process(n_processes: int, T: int, dims: tuple[int, ...]) -> int:
    """
    Estimate a safe number of pixels to allocate to each parallel process at a time
    Taken from CNMF.fit (TODO factor this out in caiman and just import it)
    """
    avail_memory_per_process = avail_bytes_per_process(n_processes) / 2.0**30
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


def make_chunk_projection_helper(args: tuple[str, slice, Optional[int], str, bool]):
    movie_path, chunk_slice, page, proj_type, ignore_nan = args
    subindices = (slice(None), slice(None), chunk_slice)
    if page is not None:
        subindices += (page,)

    mov: cm.movie = cm.load(movie_path, subindices=subindices)
    # flatten to pixels x time
    Yr = mov.reshape((mov.shape[0], -1), order='F').T
    return make_chunk_projection(Yr, proj_type, ignore_nan=ignore_nan)


def make_projection_parallel(movie_path: str, proj_type: str, dview: Optional[Cluster], ignore_nan=False) -> np.ndarray:
    """
    Compute projection in chunks that are small enough to fit in memory
    movie_path: path to movie that can be memory-mapped using caiman.load
    """
    dims, T = get_file_size(movie_path)

    # use n_pixels_per_process from CNMF to avoid running out of memory
    chunk_size = estimate_n_pixels_per_process(get_n_processes(dview), T, dims)
    chunk_columns = max(chunk_size // dims[0], 1)

    # divide movie into chunks of columns
    chunk_starts = range(0, dims[1], chunk_columns)
    chunk_slices = [
        slice(start, min(start + chunk_columns, dims[1])) for start in chunk_starts
    ]

    if len(dims) > 2 and dims[2] > 1:
        args = []
        for page in range(dims[2]):
            args.extend([
                (movie_path, chunk_slice, page, proj_type, ignore_nan)
                for chunk_slice in chunk_slices
            ])
    else:
        args = [
            (movie_path, chunk_slice, None, proj_type, ignore_nan)
            for chunk_slice in chunk_slices
        ]

    if dview is None:
        map_fn = map
    elif isinstance(dview, Pool):
        map_fn = dview.map
    else:
        map_fn = dview.map_sync

    chunk_projs = map_fn(make_chunk_projection_helper, args)
    p_img_flat = np.concatenate(list(chunk_projs), axis=0)
    return np.reshape(p_img_flat, dims, order="F")


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


ChunkDims = tuple[slice, slice]
ChunkSpec = tuple[ChunkDims, ChunkDims, ChunkDims]  # input, output, patch subinds

def make_correlation_parallel(movie_path: Union[str, Path], dview: Optional[Cluster]) -> np.ndarray:
    """
    Compute local correlations in chunks that are small enough to fit in memory
    movie_path: path to movie that can be memory-mapped using caiman.load
    """
    dims, T = get_file_size(movie_path)

    # use n_pixels_per_process from CNMF to avoid running out of memory
    chunk_size = estimate_n_pixels_per_process(get_n_processes(dview), T, dims)
    patches = make_correlation_patches(dims, chunk_size)
    
    # do correlation calculation in parallel
    args = [(str(movie_path), p[0]) for p in patches]
    if dview is None:
        map_fn = map
    elif isinstance(dview, Pool):
        map_fn = dview.map
    else:
        map_fn = dview.map_sync
    
    patch_corrs = map_fn(chunk_correlation_helper, args)
    output_img = np.empty(dims, dtype=np.float32)
    for (_, output_coords, subinds), patch_corr in zip(patches, patch_corrs):
        output_img[output_coords] = patch_corr[subinds]
    
    return output_img


def save_correlation_parallel(uuid, movie_path: Union[str, Path], output_dir: Path, dview: Optional[Cluster]) -> Path:
    """Compute and save local correlations in chunks that are small enough to fit in memory"""
    corr_img = make_correlation_parallel(movie_path, dview)
    corr_img_path = output_dir.joinpath(f"{uuid}_cn.npy")
    np.save(str(corr_img_path), corr_img, allow_pickle=False)
    return corr_img_path


def chunk_correlation_helper(args: tuple[str, ChunkDims]) -> np.ndarray:
    movie_path, dims_input = args
    mov = cm.load(movie_path, subindices=(slice(None),) + dims_input)
    return local_correlations(mov, swap_dim=False)


def make_correlation_patches(dims: tuple[int, ...], chunk_size: int) -> list[ChunkSpec]:
    """
    Compute dimensions for dividing movie (ideally C-order) into patches for correlation calculation.
    Overlap = 2 to avoid edge effects except on the edge.
    Each entry of the returned list contains 3 (Y, X) tuples of slices:
    - input coordinates (for getting sub-movie to compute correlation on)
    - output coordinates (for assigning result to full correlation image, excludes inner borders)
    - patch sub-indices (to index result for assignment to output)
    """
    window_size = math.floor(math.sqrt(chunk_size))

    # first get patch starts and sizes for each dimension
    patch_coords_y = make_correlation_patches_for_dim(dims[0], window_size)
    patch_coords_x = make_correlation_patches_for_dim(dims[1], window_size)
    return [
        ((input_y, input_x), (output_y, output_x), (subind_y, subind_x))
        for input_y, output_y, subind_y in patch_coords_y
        for input_x, output_x, subind_x in patch_coords_x
    ]


def make_correlation_patches_for_dim(dim: int, window_size: int) -> list[tuple[slice, slice, slice]]:
    """
    Like make_correlation_patches but for just one dimension
    """
    overlap = 2  # so that edge pixel in one patch is a non-edge pixel in the next
    window_size = max(window_size, overlap + 1)
    stride = window_size - overlap

    patch_starts = range(0, dim - overlap, stride)  # last <overlap> pixels are covered by last window
    patch_ends = [start + window_size for start in patch_starts[:-1]] + [dim]
    patch_coords: list[tuple[slice, slice, slice]] = []

    for start, end in zip(patch_starts, patch_ends):
        is_first = start == patch_starts[0]
        is_last = start == patch_starts[-1]
        patch_coords.append((
            slice(start, end),
            slice(start if is_first else start + 1, end if is_last else end-1),
            slice(0 if is_first else 1, None if is_last else -1)
        ))
    
    return patch_coords
