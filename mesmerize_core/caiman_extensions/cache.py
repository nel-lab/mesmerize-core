from functools import wraps
import pandas as pd
import time
import numpy as np


def check_arg_equality(args, cache_args):
    if not type(args) == type(cache_args):
        return False

    if isinstance(cache_args, np.ndarray):
        return np.array_equal(cache_args, args)
    else:
        return cache_args == args


class Cache:
    def __init__(self, cache_size=10):
        self.cache = pd.DataFrame(data=None, columns=['uuid', 'function', 'args', 'kwargs', 'return_val', 'time_stamp'])
        self.cache_size = cache_size

    def get_cache(self):
        print(self.cache)

    def clear_cache(self):
        while len(self.cache.index) != 0:
            self.cache.drop(index=self.cache.index[-1], axis=0, inplace=True)

    def set_maxsize(self, max_size: int):
        self.cache_size = max_size

    def use_cache(self, func):
        @wraps(func)
        def _use_cache(instance, *args, **kwargs):

            # if cache is empty, will always be a cache miss
            if len(self.cache.index) == 0:
                return_val = func(instance, *args, **kwargs)
                self.cache.loc[len(self.cache.index)] = [instance._series['uuid'], func.__name__, args, kwargs,
                                                         return_val, time.time()]

            # checking to see if there is a cache hit
            for i in range(len(self.cache.index)):
                if self.cache.iloc[i, 0] == instance._series['uuid'] and self.cache.iloc[i, 1] == func.__name__ and check_arg_equality(args, self.cache.iloc[i, 2]) and check_arg_equality(kwargs, self.cache.iloc[i, 3]):
                    self.cache.iloc[i, 5] = time.time()
                    return_val = self.cache.iloc[i, 4]
                    return self.cache.iloc[i, 4]

            # no cache hit, must check cache limit, and if limit is going to be exceeded...remove least recently used and add new entry
            if len(self.cache.index) == self.cache_size:
                return_val = func(instance, *args, **kwargs)
                self.cache.drop(index=self.cache.sort_values(by=['time_stamp'], ascending=False).index[-1], axis=0,
                                inplace=True)
                self.cache = self.cache.reset_index(drop=True)
                self.cache.loc[len(self.cache.index)] = [instance._series['uuid'], func.__name__, args, kwargs, return_val,
                                                         time.time()]
                return self.cache.iloc[len(self.cache.index) - 1, 4]
            else:
                return_val = func(instance, *args, **kwargs)
                self.cache.loc[len(self.cache.index)] = [instance._series['uuid'], func.__name__, args, kwargs, return_val,
                                                         time.time()]

            return return_val

        return _use_cache

