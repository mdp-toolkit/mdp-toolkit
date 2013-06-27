"""MDP extension to use theano to do dot products (matrix multiplies) on the GPU.
"""
__docformat__="restructuredtext en"

from ..utils import mult
from ..extension import ExtensionNode, activate_extension, deactivate_extension
import os

standard_mult = mult

def activate_gpu():
    """Activate theano extension.
    
    Swaps in a Theano-enabled GPU dot product instead of numx.dot for utils.mult.
    """
    # Set an environment variable before importing theano so theano knows to use the GPU
    os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32,mode=FAST_RUN'
    from theano import function, config
    import theano.tensor as T
    # Declare theano symbolic variables and function
    a, b = T.matrices('a','b')
    mult = function([a,b], T.dot(a,b),allow_input_downcast=True)
    activate_extension('gpu')
    
def deactivate_gpu():
    mult = standard_mult
    del os.environ['THEANO_FLAGS']
    del function, config, T    
    deactivate_extension('gpu')
    
class gpuify(object):
    """Context manager for the theano extension.
    """
    
    def __enter__(self):
        activate_gpu()
        
    def __exit__(self):
        deactivate_gpu()
