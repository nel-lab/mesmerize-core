import caiman as cm
from contextlib import contextmanager
from ipyparallel import DirectView
from multiprocessing.pool import Pool
import os
import psutil
from typing import Union, Optional, Generator

Cluster = Union[Pool, DirectView]

def get_n_processes(dview: Optional[Cluster]) -> int:
    """Infer number of processes in a multiprocessing or ipyparallel cluster"""
    if isinstance(dview, Pool) and hasattr(dview, '_processes'):
        return dview._processes
    elif isinstance(dview, DirectView):
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
        if "MESMERIZE_N_PROCESSES" in os.environ.keys():
            try:
                n_processes = int(os.environ["MESMERIZE_N_PROCESSES"])
            except:
                n_processes = psutil.cpu_count() - 1
        else:
            n_processes = psutil.cpu_count() - 1

        # Start cluster for parallel processing
        _, dview, n_processes = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=n_processes, single_thread=False
        )
        try:
            yield dview, n_processes
        finally:
            cm.stop_server(dview=dview)
