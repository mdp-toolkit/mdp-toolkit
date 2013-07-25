"""MDP extension that does all dot products (matrix multiplies too) on the GPU.

This extension uses theano as the interface to the GPU. Theano can be found at
https://pypi.python.org/pypi/Theano. We require version 0.6.0 at least in order
to make sure the latest GPU code is available to us.
"""
__docformat__="restructuredtext en"

import mdp.utils
from ..extension import ExtensionNode, activate_extension, deactivate_extension
from theano import function
import theano.tensor as T

# Save the default dot product behavior
standard_mult = mdp.utils.mult
# Declare theano symbolic variables and function
a, b = T.matrices('a','b')
theano_mult = function([a,b], T.dot(a,b),allow_input_downcast=True)

def activate_theano():
    """Activate theano extension.
    
    Swaps in a Theano-enabled GPU dot product instead of numx.dot for utils.mult.
    """
    
    mdp.utils.mult = theano_mult
    #activate_extension('theano')
    
def deactivate_theano():
    """De-activate theano extension."""
    mdp.utils.mult = standard_mult 
    #deactivate_extension('theano')
    
class theanoize(object):
    """Context manager for the theano extension.
    
    Because the theano extension does not use an ExtensionNode, it can only be activated
    through this custom context manager. You can use this extension with the following
    syntax:
    
    >>> with mdp.theano.theanoize():
    ...     # let 'a' and 'b' be numpy arrays of appropriate dimension
    ...     utils.mult(a,b)
    ...     # more complex MDP operations make use of utils.mult
    ...     y_pca = mdp.pca(a)
    """
    
    def __enter__(self):
        activate_theano()
        
    def __exit__(self, type, value, traceback):
        deactivate_theano()
