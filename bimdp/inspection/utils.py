"""
Some helper functions and classes for inspection.
"""
from future import standard_library
standard_library.install_aliases()
from builtins import next
from builtins import object

import os
import pickle as pickle


def robust_pickle(path, filename, obj):
    """Robust pickle function, creates path if it does not exist."""
    filename = os.path.join(path, filename)
    try:
        picke_file = open(filename, "wb")
    except IOError as inst:
        error_code = inst.args[0]
        if error_code == 2:  # path does not exist
            os.makedirs(path)
            picke_file = open(filename, "wb")
        else:
            raise
    try:
        pickle.dump(obj, picke_file, -1)
    finally:
        picke_file.close()

def robust_write_file(path, filename, content):
    """Create a file with the given content and return the filename.

    If the provided path does not exist it will be created.
    If the file already exists it will be overwritten.
    """
    try:
        new_file = open(os.path.join(path, filename), "w")
    except IOError as inst:
        error_code = inst.args[0]
        if error_code == 2:  # path does not exist
            os.makedirs(path)
            new_file = open(os.path.join(path, filename), "w")
        else:
            raise
    new_file.write(content)
    return filename

def first_iterable_elem(iterable):
    """Helper function to get the first element of an iterator or iterable.

    The return value is a tuple of the first element and the iterable.
    If the iterable is actually an iterator then a decorator is used to wrap
    it and extract the first element in a non-consuming way.
    """
    if iter(iterable) is iterable:
        # iterable is actually iterator, have to wrap it
        peek_iter = PeekIterator(iterable)
        first_elem = peek_iter.peek()
        return first_elem, peek_iter
    else:
        first_elem = next(iter(iterable))
        return first_elem, iterable


class PeekIterator(object):
    """Look-ahead iterator decorator."""

    def __init__(self, iterator):
        self.iterator = iterator
        # we simplicity we do not use collections.deque
        self.cache = []

    def peek(self):
        """Return the next element in the iterator without consuming it.

        So the returned elements will still be returned by next in the normal
        order. If the iterator has no next element then the StopIterator
        exception is passed.
        """
        next_elem = next(self)
        # TODO: use a dequeue for better efficiency
        self.cache = [next_elem] + self.cache
        return next_elem

    def __next__(self):
        if self.cache:
            return self.cache.pop()
        else:
            return next(self.iterator)

    def __iter__(self):
        return self
