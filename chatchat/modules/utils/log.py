# coding=utf-8

import logging
import time


def record_log(func, record_duration=True):
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        t0 = time.time()
        logger.info(f"{func.__name__} start.")
        result = func(*args, **kwargs)
        cost_time = time.time() - t0
        if record_duration:
            logger.info(f"{func.__name__} end in {cost_time:.2f} seconds.")
        else:
            logger.info(f"{func.__name__} end.")
        return result

    return wrapper
