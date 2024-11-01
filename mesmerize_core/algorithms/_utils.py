from pathlib import Path
import psutil
from typing import Optional, Union

import caiman as cm
from ipyparallel import DirectView
from multiprocessing.pool import Pool
import numpy as np

Cluster = Union[Pool, DirectView]

def get_n_processes(dview: Optional[Cluster]) -> int:
    """Infer number of processes in a multiprocessing or ipyparallel cluster"""
    if isinstance(dview, Pool) and hasattr(dview, '_processes'):
        return dview._processes
    elif isinstance(dview, DirectView):
        return len(dview)
    else:
        return 1


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

def make_chunk_projection_helper(args: tuple[str, slice, str]):
    Yr_name, chunk_slice, proj_type = args
    Yr, _, _ = cm.load_memmap(Yr_name)
    return make_chunk_projection(Yr[chunk_slice], proj_type)


def make_projection_parallel(movie_path: str, proj_type: str, dview: Optional[Cluster]) -> np.ndarray:
    if dview is None:
        p_img_flat = make_chunk_projection(Yr, proj_type)
    else:
        # use n_pixels_per_process from CNMF to avoid running out of memory
        Yr, dims, T = cm.load_memmap(movie_path)
        n_pix = Yr.shape[0]
        chunk_size = estimate_n_pixels_per_process(get_n_processes(dview), T, dims)
        chunk_starts = range(0, n_pix, chunk_size)
        chunk_slices = [slice(start, min(start + chunk_size, n_pix)) for start in chunk_starts]
        args = [(movie_path, chunk_slice, proj_type) for chunk_slice in chunk_slices]
        map_fn = dview.map if isinstance(dview, Pool) else dview.map_sync
        chunk_projs = map_fn(make_chunk_projection_helper, args)
        p_img_flat = np.concatenate(chunk_projs, axis=0)
    return np.reshape(p_img_flat, dims, order='F')


def save_projections_parallel(uuid, movie_path: Union[str, Path], output_dir: Path, dview: Optional[Cluster]
                              ) -> dict[str, Path]:
    proj_paths = dict()
    for proj_type in ["mean", "std", "max"]:
        p_img = make_projection_parallel(str(movie_path), "nan" + proj_type, dview=dview)
        proj_paths[proj_type] = output_dir.joinpath(
            f"{uuid}_{proj_type}_projection.npy"
        )
        np.save(str(proj_paths[proj_type]), p_img)
    return proj_paths
