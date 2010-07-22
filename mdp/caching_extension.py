
# TODO: decorator per le singole istanze

# TODO: settare la directory di cache manualmente o a caso

import joblib
from copy import deepcopy
from .extension import ExtensionNode
from .signal_node import Node

CACHEDIR = 'c:/sys/cygwin/home/berkes/del/joblibcache/'

def activate_caching():
    pass

class CacheExecuteExtensionNode(ExtensionNode, Node):
    """MDP Extension for caching execution results.

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
            self._memory = joblib.Memory(CACHEDIR)
            self._cached_execute = self._memory.cache(
                self._non_extension_execute.im_func)
        return self._cached_execute(self, x)
