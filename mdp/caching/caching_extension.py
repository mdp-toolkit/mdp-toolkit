"""MDP extension to cache the execution phase of nodes.

This extension is based on the 'joblib' library by Gael Varoquaux,
available at http://packages.python.org/joblib/ . At the moment, this
extension supports joblib v. 0.4.6 .
"""

import joblib
from copy import deepcopy
from tempfile import mkdtemp

from ..extension import ExtensionNode, activate_extension, deactivate_extension
from ..signal_node import Node

# FIXME: if cachedir is modified after an instance has been cached,
# the extension does not notice and keeps caching in the old one

# TODO: the latest version of joblib fixes a bug that did not allow it
# to decorate methods; it should now be possible to remove the
# __getstate__ method; UPDATE Dec 8 2010: this fix is only in the
# development branch, it has not been fixed os as of joblib 0.4.6


_cachedir = None
_memory = None

def set_cachedir(cachedir=None, verbose=0):
    """Set root directory for the joblib cache.

    cachedir -- the cache directory name; if None, a temporary directory
                is created using tempfile.mkdtemp()
    verbose -- an integer number, control the verbosity of the cache
    """

    global _cachedir
    global _memory

    if cachedir is None:
        cachedir = mkdtemp()
    _cachedir = cachedir
    _memory = joblib.Memory(cachedir, verbose=0)

# initialize cache with temporary directory
set_cachedir()

class CacheExecuteExtensionNode(ExtensionNode, Node):
    """MDP extension for caching execution results.

    Activating this extension results in all nodes caching the return
    values of the 'execute' methods.

    Warning: this extension might brake the algorithms if nodes rely
    on side effects.
    """

    extension_name = 'cache_execute'

    def __getstate__(self):
        # This function is used by Pickler to decide what to
        # pickle. We hide self._cached_execute to avoid it complaining
        # when computing the joblib hash number
        dct = self.__dict__
        if not dct.has_key('_cached_execute'):
            return dct
        else:
            dct = deepcopy(dct)
            del dct['_cached_execute']
            return dct

    _CLASS_CACHING = False

    def set_cache(self, active=True):
        self._cache_active = active

    def execute(self, x, *args, **kwargs):
        if not hasattr(self, '_cached_execute'):
            global _memory
            self._cached_execute = _memory.cache(
                self._non_extension_execute.im_func)
        return self._cached_execute(self, x)


# ------- helper functions and context manager

def activate_caching(cachedir=None, verbose=False):
    """Activate caching extension.

    cachedir -- The root of the joblib cache, or a temporary directory if None.
    """
    set_cachedir(cachedir)
    activate_extension('cache_execute')

def deactivate_caching(cachedir=None):
    """De-activate caching extension."""
    deactivate_extension('cache_execute')

class cache(object):
    """Context manager for the 'cache_execute' extension.

    This allows using the caching extension using a 'with'
    statement, as in:

    with mdp.caching.cache(CACHEDIR):
        # 'node' is executed caching the results in CACHEDIR
        node.execute(x)

    If the argument to the context manager is not specified, caching is
    done in a temporary directory.
    """

    def __init__(self, cachedir=None):
        self.cachedir = cachedir

    def __enter__(self):
        activate_caching(self.cachedir)

    def __exit__(self, type, value, traceback):
        deactivate_caching()
