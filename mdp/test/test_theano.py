"""Test gputheano extension."""
import theano
from _tools import *

def test_mult_remapping():
    mdp.gputheano.activate_theano()
    assert type(mdp.utils.mult) == theano.compile.function_module.Function
    mdp.gputheano.deactivate_theano()
    assert mdp.utils.mult == np.dot
