"""MDP extension to use theano to do dot products (matrix multiplies) on the GPU.
"""
__docformat__="restructuredtext en"

from ..utils import mult
from ..extension import ExtensionNode, activate_extension, deactivate_extension
from theano import function
import theano.tensor as T

standard_mult = mult

def activate_gpu():
    """Activate theano extension.
    
    Swaps in a Theano-enabled GPU dot product instead of numx.dot for utils.mult.
    """
    
    # Declare theano symbolic variables and function
    a, b = T.matrices('a','b')
    mult = function([a,b], T.dot(a,b),allow_input_downcast=True)
    #activate_extension('gpu')
    
def deactivate_gpu():
    mult = standard_mult 
    #deactivate_extension('gpu')
    
class gpuify(object):
    """Context manager for the theano extension.
    """
    
    def __enter__(self):
        activate_gpu()
        
    def __exit__(self, type, value, traceback):
        deactivate_gpu()
