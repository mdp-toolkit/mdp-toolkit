"""MDP extension that does all dot products (matrix multiplies too) on the GPU.

This extension uses theano as the interface to the GPU. Theano can be found at
https://pypi.python.org/pypi/Theano. We require version 0.6.0 at least in order
to make sure the latest GPU code is available to us.
"""
__docformat__="restructuredtext en"

from ..utils import mult
from ..extension import ExtensionNode, activate_extension, deactivate_extension
from theano import function
import theano.tensor as T

# Save the default dot product behavior
standard_mult = mult
# Declare theano symbolic variables and function
a, b = T.matrices('a','b')
gpu_mult = function([a,b], T.dot(a,b),allow_input_downcast=True)

def activate_gpu():
    """Activate gpu extension.
    
    Swaps in a Theano-enabled GPU dot product instead of numx.dot for utils.mult.
    """
    
    mult = gpu_mult
    #activate_extension('gpu')
    
def deactivate_gpu():
    """De-activate gpu extension."""
    mult = standard_mult 
    #deactivate_extension('gpu')
    
class gpuify(object):
    """Context manager for the 'gpu' extension.
    
    Because the gpu extension does not use an ExtensionNode, it can only be activated
    through this custom context manager. You can use this extension with the following
    syntax:
    
    >>> with mdp.gpu.gpuify():
    ...     # let 'a' and 'b' be numpy arrays of appropriate dimension
    ...     utils.mult(a,b)
    ...     # more complex MDP operations make use of utils.mult
    ...     y_pca = mdp.pca(a)
    """
    
    def __enter__(self):
        activate_gpu()
        
    def __exit__(self, type, value, traceback):
        deactivate_gpu()
