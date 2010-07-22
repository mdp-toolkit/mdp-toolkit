
# TODO: decorator per le singole istanze

# TODO: docs

import joblib
from copy import deepcopy
from tempfile import mkdtemp

from ..extension import ExtensionNode, activate_extension, deactivate_extension
from ..signal_node import Node

_cachedir = None
_memory = None

def set_cachedir(cachedir=None):
    """Set root directory for the joblib cache.

    If the input argument is None, a temporary directory is created using
    tempfile.mkdtemp()."""

    global _cachedir
    global _memory
    
    if cachedir is None:
        cachedir = mkdtemp()
    _cachedir = cachedir
    _memory = joblib.Memory(cachedir)

# initialize cache with temporary variable
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
        dct = self.__dict__
        if not dct.has_key('_cached_execute'):
            return dct
        else:
            dct = deepcopy(dct)
            del dct['_cached_execute']
            return dct

    def execute(self, x, *args, **kwargs):
        if not hasattr(self, '_memory'):
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

    This allows you to use the caching extension using a 'with'
    statement, as in:

    with mdp.caching(CACHEDIR):
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
