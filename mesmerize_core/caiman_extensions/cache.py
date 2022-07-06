from functools import wraps
from typing import Union, Optional, Tuple
from builtins import list

import pandas as pd
import time
import numpy as np
import sys
from pathlib import Path
from caiman.source_extraction.cnmf import CNMF
import re


def _check_arg_equality(args, cache_args):
    if not type(args) == type(cache_args):
        return False
    if isinstance(cache_args, np.ndarray):
        return np.array_equal(cache_args, args)
    else:
        return cache_args == args


def _check_args_equality(args, cache_args):
    equality = list()
    if isinstance(args, tuple):
        for arg, cache_arg in zip(args, cache_args):
            equality.append(_check_arg_equality(arg, cache_arg))
    else:
        for k in args.keys():
            equality.append(_check_arg_equality(args[k], cache_args[k]))
    return all(equality)


class Cache:
    def __init__(self, cache_size: Optional[Union[int, str]] = None):
        self.cache = pd.DataFrame(
            data=None,
            columns=["uuid", "function", "args", "kwargs", "return_val", "time_stamp"],
        )
        if cache_size is None:
            self.size = '1G'
            self.storage_type = 'RAM'
        elif isinstance(cache_size, int):
            self.storage_type = 'ITEMS'
            self.size = cache_size
        else:
            self.storage_type = 'RAM'
            self.size = cache_size

    def get_cache(self):
        print(self.cache)

    def clear_cache(self):
        while len(self.cache.index) != 0:
            self.cache.drop(index=self.cache.index[-1], axis=0, inplace=True)

    def set_maxsize(self, max_size: Union[int, str]):
        if isinstance(max_size, str):
            self.storage_type = 'RAM'
            self.size = max_size
        else:
            self.storage_type = 'ITEMS'
            self.size = max_size

    def _get_cache_size_bytes(self):
        """Returns in GiB or MB"""
        cache_size = 0
        for i in range(len(self.cache.index)):
            if isinstance(self.cache.iloc[i, 4], list):
                for array in self.cache.iloc[i, 4]:
                    cache_size += array.data.nbytes
            if isinstance(self.cache.iloc[i, 4], np.ndarray):
                cache_size += self.cache.iloc[i, 4].data.nbytes
            elif isinstance(self.cache.iloc[i, 4], tuple):
                for array in self.cache.iloc[i, 4]:
                    cache_size += array.data.nbytes
            elif isinstance(self.cache.iloc[i, 4], Path):
                cache_size += 0
            elif isinstance(self.cache.iloc[i, 4], CNMF):
                cache_size += (self.cache.iloc[i, 4].estimates.A.data.nbytes + self.cache.iloc[i, 4].estimates.C.data.nbytes + self.cache.iloc[i, 4].estimates.b.data.nbytes + self.cache.iloc[i, 4].estimates.f.data.nbytes)
            else:
                cache_size += sys.getsizeof(self.cache.iloc[i, 4])

        if self.size.endswith('G'):
            cache_size = cache_size / 1024**3
        elif self.size.endswith('M'):
            cache_size = cache_size / 1024**2
        return cache_size

    def use_cache(self, func):
        @wraps(func)
        def _use_cache(instance, *args, **kwargs):

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

            # checking to see if there is a cache hit
            for i in range(len(self.cache.index)):
                if (
                        self.cache.iloc[i, 0] == instance._series["uuid"]
                        and self.cache.iloc[i, 1] == func.__name__
                        and _check_args_equality(args, self.cache.iloc[i, 2])
                        and _check_arg_equality(kwargs, self.cache.iloc[i, 3])
                ):
                    self.cache.iloc[i, 5] = time.time()
                    return_val = self.cache.iloc[i, 4]
                    return self.cache.iloc[i, 4]

            # no cache hit, must check cache limit, and if limit is going to be exceeded...remove least recently used and add new entry
            # if memory type is 'ITEMS': drop the least recently used and then add new item
            if self.storage_type == 'ITEMS' and len(self.cache.index) >= self.size:
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
                return self.cache.iloc[len(self.cache.index) - 1, 4]
            # if memory type is 'RAM': add new item and then remove least recently used items until cache is under correct size again
            elif self.storage_type == 'RAM' and self._get_cache_size_bytes() > int(re.split('[a-zA-Z]', self.size)[0]):
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

            return return_val

        return _use_cache
