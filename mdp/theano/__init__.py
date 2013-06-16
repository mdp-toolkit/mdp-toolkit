from theano_extension import (activate_caching, deactivate_caching, __doc__, __docformat__)

from mdp.utils import fixup_namespace

__all__ = ['activate_theano', 'deactivate_theano']

fixup_namespace(__name__,__all__,('theano_extension','fixup_namespace',))
