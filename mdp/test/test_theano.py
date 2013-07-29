"""Test gputheano extension."""
import theano
from _tools import *

requires_theano = skip_on_condition(
    "not mdp.config.has_theano",
    "This test requires the 'theano' module.")

@requires_theano
def test_mult_remapping():
    mdp.gputheano.activate_theano()
    assert type(mdp.utils.mult) == theano.compile.function_module.Function
    mdp.gputheano.deactivate_theano()
    assert mdp.utils.mult == numx.dot

@requires_theano
def test_random_matrix_mult():
    random_matrix_1 = numx_rand.rand(4,4)
    random_matrix_2 = numx_rand.rand(4,4)
    cpu_result = mdp.utils.mult(random_matrix_1, random_matrix_2)
    with mdp.gputheano.theanoize():
        gpu_result = mdp.utils.mult(random_matrix_1, random_matrix_2)
    assert_array_almost_equal_diff(cpu_result, gpu_result, 12, 
                                   'GPU not equal CPU result to 12 digits.')
