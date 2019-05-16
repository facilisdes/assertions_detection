import functools
import time
# from common.debug import *

def execTimer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def log(func):
    """logs function's output"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        value = func(*args, **kwargs)
        print(f"Finished {func.__name__!r} with {value}")
        return value
    return wrapper_timer
