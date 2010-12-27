from caching_extension import (activate_caching, deactivate_caching,
                               cache, set_cachedir,
                               __doc__, __docformat__)

del caching_extension
__all__ = ['activate_caching', 'deactivate_caching',
           'cache', 'set_cachedir']
