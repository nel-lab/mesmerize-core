from functools import wraps
from typing import Union
from uuid import UUID
import pandas as pd

from mesmerize_core.caiman_extensions._batch_exceptions import BatchItemNotRunError, BatchItemUnsuccessfulError, \
    WrongAlgorithmExtensionError


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
    """
    Parses uuid identifier that can be passed in various ways and returns it as a UUID string regardless of input type.
    """
    @wraps(func)
    def _parser(instance, *args, **kwargs):
        if "identifier" in kwargs.keys():
            u: Union[int, str, UUID] = kwargs["index"]
        elif len(args) > 0:
            u = args[0]  # always first positional arg

        if not isinstance(u, (pd.Series, UUID, str)):
            raise TypeError(
                "Passed index must be one of the following types:\n"
                "`pandas.Series`, `UUID`, `str`"
            )

        # if the batch item itself was passed
        if isinstance(u, pd.Series):
            u = u["uuid"]

        # if the passed `index` is already a UUID
        if isinstance(u, (UUID, str)):
            _index = instance._df[instance._df["uuid"] == str(u)].index

            # make sure it exists in the dataframe
            if _index.size == 0:
                raise ValueError(f"No batch item found with uuid: {u}")

            u = str(u)

        if "identifier" in kwargs.keys():
            kwargs["identifier"] = u
        else:
            args = (u, *args[1:])

        return func(instance, *args, **kwargs)
    return _parser
