from contextlib import contextmanager
import os
import psutil
from typing import (Optional, Union, Generator, Protocol,
                    Callable, TypeVar, Sequence, Iterable, runtime_checkable)

import caiman as cm
from caiman.cluster import setup_cluster
from multiprocessing.pool import Pool


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


Cluster = Union[Pool, CustomCluster]


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
