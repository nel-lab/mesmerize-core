from functools import wraps
from typing import Union, Optional

import pandas as pd
import time
import numpy as np
import sys
from caiman.source_extraction.cnmf import CNMF
import re
from sys import getsizeof
import copy


def _check_arg_equality(args, cache_args):
    if not type(args) == type(cache_args):
        return False
    if isinstance(cache_args, np.ndarray):
        return np.array_equal(cache_args, args)
    else:
        return cache_args == args


def _check_args_equality(args, cache_args):
    if len(args) != len(cache_args):
        return False
    equality = list()
    if isinstance(args, tuple):
        for arg, cache_arg in zip(args, cache_args):
            equality.append(_check_arg_equality(arg, cache_arg))
    else:
        for k in args.keys():
            equality.append(_check_arg_equality(args[k], cache_args[k]))
    return all(equality)


def _return_wrapper(output, copy_bool):
    if copy_bool == True:
        return copy.deepcopy(output)
    else:
        return output


class Cache:
    def __init__(self, cache_size: Optional[Union[int, str]] = None):
        self.cache = pd.DataFrame(
            data=None,
            columns=["uuid", "function", "args", "kwargs", "return_val", "time_stamp"],
        )
        self.set_maxsize(cache_size)

    def get_cache(self):
        return self.cache

    def clear_cache(self):
        while len(self.cache.index) != 0:
            self.cache.drop(index=self.cache.index[-1], axis=0, inplace=True)

    def set_maxsize(self, max_size: Union[int, str]):
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
        """Returns in bytes"""
        cache_size = 0
        for i in range(len(self.cache.index)):
            if isinstance(self.cache.iloc[i, 4], np.ndarray):
                cache_size += self.cache.iloc[i, 4].data.nbytes
            elif isinstance(self.cache.iloc[i, 4], (tuple, list)):
                for lists in self.cache.iloc[i, 4]:
                    for array in lists:
                        cache_size += array.data.nbytes
            elif isinstance(self.cache.iloc[i, 4], CNMF):
                sizes = list()
                for attr in self.cache.iloc[i, 4].estimates.__dict__.values():
                    if isinstance(attr, np.ndarray):
                        sizes.append(attr.data.nbytes)
                    else:
                        sizes.append(getsizeof(attr))
            else:
                cache_size += sys.getsizeof(self.cache.iloc[i, 4])

        return cache_size

    def use_cache(self, func):
        @wraps(func)
        def _use_cache(instance, *args, **kwargs):
            if "return_copy" in kwargs.keys():
                return_copy = kwargs["return_copy"]
            else:
                return_copy = True

            if self.size == 0:
                self.clear_cache()
                return _return_wrapper(func(instance, *args, **kwargs), return_copy)

            # if cache is empty, will always be a cache miss
            if len(self.cache.index) == 0:
                return_val = func(instance, *args, **kwargs)
                self.cache.loc[len(self.cache.index)] = [
                    instance._series["uuid"],
                    func.__name__,
                    args,
                    kwargs,
                    return_val,
                    time.time(),
                ]
                return _return_wrapper(return_val, copy_bool=return_copy)

            # checking to see if there is a cache hit
            for i in range(len(self.cache.index)):
                if (
                    self.cache.iloc[i, 0] == instance._series["uuid"]
                    and self.cache.iloc[i, 1] == func.__name__
                    and _check_args_equality(args, self.cache.iloc[i, 2])
                    and _check_args_equality(kwargs, self.cache.iloc[i, 3])
                ):
                    self.cache.iloc[i, 5] = time.time()
                    return_val = self.cache.iloc[i, 4]
                    return _return_wrapper(self.cache.iloc[i, 4], copy_bool=return_copy)

            # no cache hit, must check cache limit, and if limit is going to be exceeded...remove least recently used and add new entry
            # if memory type is 'ITEMS': drop the least recently used and then add new item
            if self.storage_type == "ITEMS" and len(self.cache.index) >= self.size:
                return_val = func(instance, *args, **kwargs)
                self.cache.drop(
                    index=self.cache.sort_values(
                        by=["time_stamp"], ascending=False
                    ).index[-1],
                    axis=0,
                    inplace=True,
                )
                self.cache = self.cache.reset_index(drop=True)
                self.cache.loc[len(self.cache.index)] = [
                    instance._series["uuid"],
                    func.__name__,
                    args,
                    kwargs,
                    return_val,
                    time.time(),
                ]
                return _return_wrapper(
                    self.cache.iloc[len(self.cache.index) - 1, 4], copy_bool=return_copy
                )
            # if memory type is 'RAM': add new item and then remove least recently used items until cache is under correct size again
            elif self.storage_type == "RAM":
                while self._get_cache_size_bytes() > self.size:
                    self.cache.drop(
                        index=self.cache.sort_values(
                            by=["time_stamp"], ascending=False
                        ).index[-1],
                        axis=0,
                        inplace=True,
                    )
                    self.cache = self.cache.reset_index(drop=True)
                return_val = func(instance, *args, **kwargs)
                self.cache.loc[len(self.cache.index)] = [
                    instance._series["uuid"],
                    func.__name__,
                    args,
                    kwargs,
                    return_val,
                    time.time(),
                ]
            # no matter the storage type if size is not going to be exceeded for either, then item can just be added to cache
            else:
                return_val = func(instance, *args, **kwargs)
                self.cache.loc[len(self.cache.index)] = [
                    instance._series["uuid"],
                    func.__name__,
                    args,
                    kwargs,
                    return_val,
                    time.time(),
                ]

            return _return_wrapper(return_val, copy_bool=return_copy)

        return _use_cache

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
            @wraps(func)
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
