from time import time


def time_decorator(func):
    def run_func_and_calculate_time(*args, **kwargs):
        start_time = time()
        func(*args, **kwargs)
        work_time = time() - start_time

    return run_func_and_calculate_time
