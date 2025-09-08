import inspect
from typing import Union, Optional, TypeVar, Callable, ParamSpec, Concatenate, cast

import pandas as pd
import time
import numpy as np
import sys
from caiman.source_extraction.cnmf import CNMF
import copy

from ..utils import wrapsmethod
from ._utils import SeriesExtensions


# type vars for decorated methods
S = TypeVar("S", bound=SeriesExtensions)
P = ParamSpec("P")
R = TypeVar("R")


def _check_arg_equality(args, cache_args) -> bool:
    if not type(args) == type(cache_args):
        return False
    if isinstance(cache_args, np.ndarray):
        return np.array_equal(cache_args, args)
    else:
        return cache_args == args


def _check_args_equality(args, cache_args) -> bool:
    if len(args) != len(cache_args):
        return False

    if isinstance(args, tuple):
        for arg, cache_arg in zip(args, cache_args):
            if not _check_arg_equality(arg, cache_arg):
                return False
    else:
        for k, v in args.items():
            if k not in cache_args or not _check_arg_equality(v, cache_args[k]):
                return False
    return True


def _return_wrapper(output: R, copy_bool: bool) -> R:
    if copy_bool == True:
        return copy.deepcopy(output)
    else:
        return output


def _get_item_size(item) -> int:
    """Recursively compute size of return value"""
    if isinstance(item, np.ndarray):
        return item.data.nbytes

    elif isinstance(item, (tuple, list)):
        size = 0
        for entry in item:
            size += _get_item_size(entry)
        return size

    elif isinstance(item, CNMF):
        size = 0
        for attr in item.estimates.__dict__.values():
            size += _get_item_size(attr)
        return size

    else:
        return sys.getsizeof(item)


class Cache:
    def __init__(self, cache_size: Optional[Union[int, str]] = None):
        self.cache = pd.DataFrame(
            data=None,
            columns=["uuid", "function", "kwargs", "return_val", "time_stamp", "added_time", "bytes"],
        )
        self.set_maxsize(cache_size)

    def get_cache(self):
        return self.cache

    def clear_cache(self):
        while len(self.cache) != 0:
            self.cache.drop(index=self.cache.index[-1], axis=0, inplace=True)

    def set_maxsize(self, max_size: Optional[Union[int, str]]):
        if max_size is None:
            self.storage_type = "RAM"
            self.size = 1024**3
        elif isinstance(max_size, str):
            self.storage_type = "RAM"
            if max_size.endswith("G"):
                self.size = int(max_size[:-1]) * 1024**3
            elif max_size.endswith("M"):
                self.size = int(max_size[:-1]) * 1024**2
        else:
            self.storage_type = "ITEMS"
            self.size = max_size

    def _get_cache_size_bytes(self):
        return self.cache.loc[:, "bytes"].sum()

    def use_cache(self, _func: Optional[Callable[Concatenate[S,P], R]] = None, *, return_copy: Optional[bool] = None):
        """
        Caching decorator.
        
        Usage:
        
        .. code-block:: python
            @cache.use_cache
            def my_costly_method(self, *, return_copy=True):
                ...
        
        Or:

        .. code-block:: python
            @cache.use_cache(return_copy=False):
            def my_method_no_copying(self):
                ...
        
        return_copy determines whether an entry that is found in the cache is copied before it is returned.
          - When there is no return_copy argument to the decorator (or when set to None), the decorated function
            *must* take return_copy as a keyword-only paramter, and this will be read by the decorator.
          - When return_copy is provided to the decorator, this value is always used for return_copy.
            The wrapped function *must not* have an argument named return_copy. 
        """
        if _func is not None:  # used as decorator directly without being called 
            return self.use_cache(return_copy=return_copy)(_func)

        def _use_cache_inner(func: Callable[Concatenate[S, P], R]):
            # get default value of return_copy from function signature
            params = inspect.signature(func).parameters
            return_copy_arg = params.get("return_copy")
            if return_copy is not None:
                if return_copy_arg is not None:
                    raise TypeError("return_copy cannot be in wrapped function signature when provided to decorator")
                return_copy_default = return_copy
            else:
                if return_copy_arg is None:
                    raise TypeError("return_copy must be in wrapped function signature when not provided to decorator")
                elif return_copy_arg.kind != inspect.Parameter.KEYWORD_ONLY:
                    raise TypeError("return_copy must be a keyword-only argument")
                elif return_copy_arg.default == inspect.Parameter.empty:
                    return_copy_default = None  # unlikely but in this case return_copy would be required
                else:
                    return_copy_default = return_copy_arg.default
                    assert isinstance(return_copy_default, bool), "return_copy default should be bool"

            @wrapsmethod(func)
            def _use_cache_wrapper(instance: S, *args: P.args, **kwargs: P.kwargs) -> R:
                # if we are not storing anything in the cache, just do the function call, no copy needed
                if self.size == 0: 
                    self.clear_cache()
                    return func(instance, *args, **kwargs)

                # iterate through signature and make dict containing arguments to compare, including defaults
                args_dict = {}
                for i, (param_name, param) in enumerate(params.items()):
                    if i == 0 or param_name == "return_copy":
                        continue  # skip self/instance and return_copy
                    elif i-1 < len(args):
                        args_dict[param_name] = args[i-1]
                    elif param_name in kwargs:
                        args_dict[param_name] = kwargs[param_name]
                    else: 
                        assert param.default != inspect.Parameter.empty, "must have a default argument or there would be a TypeError"
                        args_dict[param_name] = param.default

                # extract return_copy; return_copy is keyword only, so only have to look in kwargs
                copy_bool = return_copy  # change name to avoid assigning to parameter
                if copy_bool is None:
                    copy_bool = kwargs.get("return_copy", return_copy_default)
                    if copy_bool is None:  # no default case
                        raise TypeError("Must provide a value for return_copy")

                if not isinstance(copy_bool, bool):
                    raise TypeError("return_copy must be a bool")
            
                # checking to see if there is a cache hit
                for ind, row in self.cache.iterrows():
                    if (
                        row.at["uuid"] == instance._series["uuid"]
                        and row.at["function"] == func.__name__
                        and _check_args_equality(args_dict, row.at["kwargs"])
                    ):
                        self.cache.at[ind, "time_stamp"] = time.time()  # not supposed to modify row from iterrows
                        return_val = cast(R, row.at["return_val"])
                        return _return_wrapper(return_val, copy_bool=copy_bool)

                # no cache hit, must check cache limit, and if limit is going to be exceeded...remove least recently used and add new entry
                # if memory type is 'ITEMS': drop the least recently used and then add new item
                return_val = func(instance, *args, **kwargs)
                curr_val_size = _get_item_size(return_val)

                if self.storage_type == "ITEMS" and len(self.cache) >= self.size:
                    self.cache.drop(
                        index=self.cache.sort_values(
                            by=["time_stamp"], ascending=False
                        ).index[-1],
                        axis=0,
                        inplace=True,
                    )
                    self.cache.reset_index(drop=True, inplace=True)

                # if memory type is 'RAM': add new item and then remove least recently used items until cache is under correct size again
                elif self.storage_type == "RAM":
                    while len(self.cache) > 1 and self._get_cache_size_bytes() + curr_val_size > self.size:  # can't do anything if it's empty
                        self.cache.drop(
                            index=self.cache.sort_values(
                                by=["time_stamp"], ascending=False
                            ).index[-1],
                            axis=0,
                            inplace=True,
                        )
                        self.cache.reset_index(drop=True, inplace=True)

                # now ready to add to cache
                add_time = time.time()
                self.cache.loc[len(self.cache)] = [
                    instance._series["uuid"],
                    func.__name__,
                    args_dict,
                    return_val,
                    add_time,
                    add_time,
                    curr_val_size,
                ]
                return _return_wrapper(return_val, copy_bool=copy_bool)

            return _use_cache_wrapper
        return _use_cache_inner

    def invalidate(self, pre: bool = True, post: bool = True):
        """
        invalidate all cache entries associated to a single batch item

        Parameters
        ----------
        pre: bool
            invalidate before the decorated function has been fun

        post: bool
            invalidate after the decorated function has been fun

        """

        def _invalidate(func):
            @wrapsmethod(func)
            def __invalidate(instance, *args, **kwargs):
                u = instance._series["uuid"]

                if pre:
                    self.cache.drop(
                        self.cache.loc[self.cache["uuid"] == u].index, inplace=True
                    )

                rval = func(instance, *args, **kwargs)

                if post:
                    self.cache.drop(
                        self.cache.loc[self.cache["uuid"] == u].index, inplace=True
                    )

                return rval

            return __invalidate

        return _invalidate
