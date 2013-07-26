"""Test gputheano extension."""
import theano

def test_activate_deactivate():
    mdp.gputheano.activate_theano()
    assert 'theano' in mdp.get_active_extensions()
    mdp.gputheano.deactivate_theano()
    assert 'theano' not in mdp.get_active_extensions()

def test_mult_remapping():
    mdp.gputheano.activate_theano()
    assert type(mdp.utils.mult) == theano.compile.function_module.Function
    mdp.gputheano.deactivate_theano()
    assert mdp.utils.mult == np.dot
