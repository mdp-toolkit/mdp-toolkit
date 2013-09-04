"""MDP extension that does all dot products (matrix multiplies too) on the GPU.

This extension uses theano as the interface to the GPU. Theano can be found at
https://pypi.python.org/pypi/Theano. We require version 0.6.0 at least in order
to make sure the latest GPU code is available to us.
"""

import theano
import mdp

# save the default dot product behavior
_standard_mult = None


@mdp.extension_setup("gputheano")
def _activate_theano():
    """Activate theano extension.
    
    Swaps in a Theano-enabled GPU dot product instead of numx.dot for
    utils.mult.
    """
    global _standard_mult
    _standard_mult = mdp.utils.mult
    # declare theano symbolic variables and function
    a, b = theano.tensor.matrices('a','b')
    gputheano = theano.function([a,b], theano.tensor.dot(a,b),
                                allow_input_downcast=True)
    mdp.utils.mult = gputheano


@mdp.extension_teardown("gputheano")
def _deactivate_theano():
    """De-activate theano extension."""
    mdp.utils.mult = _standard_mult 

