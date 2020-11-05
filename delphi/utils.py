import hashlib
import multiprocessing as mp
import os
import re
import threading
from functools import wraps
from queue import Queue
from typing import Union, Iterable, Any, TypeVar
from logzero import logger

T = TypeVar('T')

yaml_path_matcher = re.compile(r'\$\{([^}^{]+)\}')


def yaml_path_constructor(loader, node):
    ''' Extract the matched value, expand env variable, and replace the match '''
    value = node.value
    match = yaml_path_matcher.match(value)
    env_var = match.group()[2:-1]
    return str(os.environ.get(env_var)) + str(value[match.end():])


def bounded_iter(iterable: Iterable[T], semaphore: threading.Semaphore) -> Iterable[T]:
    for item in iterable:
        semaphore.acquire()
        yield item


def get_example_key(content) -> str:
    return hashlib.sha1(content).hexdigest() + '.jpg'


def to_iter(queue: Union[Queue, mp.Queue]) -> Iterable[Any]:
    def iterate():
        while True:
            example = queue.get()
            if example is None:
                break
            yield example

    return iterate()


def log_exceptions(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e

    return func_wrapper
