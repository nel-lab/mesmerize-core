from functools import wraps
from typing import Union
from uuid import UUID

from mesmerize_core.caiman_extensions._batch_exceptions import BatchItemNotRunError, BatchItemUnsuccessfulError, \
    WrongAlgorithmExtensionError, PreventOverwriteError


def validate(algo: str = None):
    def dec(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._series["outputs"] is None:
                raise BatchItemNotRunError("Item has not been run")

            if algo is not None:
                if algo not in self._series["algo"]:
                    raise WrongAlgorithmExtensionError(
                        f"<{algo}> extension called for a <{self._series.algo}> item"
                    )

            if not self._series["outputs"]["success"]:
                tb = self._series["outputs"]["traceback"]
                raise BatchItemUnsuccessfulError(f"Batch item was unsuccessful, traceback from subprocess:\n{tb}")
            return func(self, *args, **kwargs)

        return wrapper

    return dec


def _verify_and_lock_batch_file(func):
    """Acquires lock and ensures batch file has the same items as current df before calling wrapped function"""
    @wraps(func)
    def wrapper(instance, *args, **kwargs):
        with instance._batch_lock:
            disk_df = instance.reload_from_disk()
            # check whether all the same UUIDs are present with the same indices
            if not instance._df["uuid"].equals(disk_df["uuid"]):
                raise PreventOverwriteError("Items on disk do not match current DataFrame; reload to synchronize")
            return func(instance, *args, **kwargs)
    return wrapper


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
