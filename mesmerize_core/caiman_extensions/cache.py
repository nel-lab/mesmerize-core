from functools import wraps
import pandas as pd
import time

class Cache:
    def __init__(self, cache_size=3):
        self.cache = pd.DataFrame(data=None, columns=['function', 'args', 'kwargs', 'return_val', 'time_stamp'])
        self.cache_size = cache_size

    def get_cache(self):
        print(self.cache)

    def use_cache(self, func):
        @wraps(func)
        def _use_cache(*args, **kwargs):

            # if cache is empty, will always be a cache miss
            if len(self.cache.index) == 0:
                self.cache.loc[len(self.cache.index)] = [func.__name__, args, kwargs, func(args, kwargs), time.time()]

            # checking to see if there is a cache hit
            for i in range(len(self.cache.index)):
                if self.cache.iloc[i, 0] == func.__name__ and self.cache.iloc[i, 1] == args and self.cache.iloc[
                    i, 2] == kwargs:
                    self.cache.iloc[i, 4] = time.time()
                    return self.cache.iloc[i, 3]

            # no cache hit, must check cache limit, and if limit is going to be exceeded...remove least recently used and add new entry
            if len(self.cache.index) == self.cache_size:
                self.cache.drop(index=self.cache.sort_values(by=['time_stamp'], ascending=False).index[-1], axis=0,
                                inplace=True)
                self.cache = self.cache.reset_index(drop=True)
                self.cache.loc[len(self.cache.index)] = [func.__name__, args, kwargs, func(args, kwargs), time.time()]
                return self.cache.iloc[len(self.cache.index) - 1, 3]
            else:
                self.cache.loc[len(self.cache.index)] = [func.__name__, args, kwargs, func(args, kwargs), time.time()]

            return func(*args, **kwargs)

        return _use_cache


cache = Cache()