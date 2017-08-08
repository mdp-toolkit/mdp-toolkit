
from ._tools import *


# TODO Need to add more test functions.
def test_basisfn_node():

    # test indicator fn
    lims = [[0], [10]]
    bfn = mdp.nodes.BasisFunctionNode(basis_name='indicator', lims=lims)

    for i in xrange(lims[1][0]):
        inp = mdp.numx.array([[i]]).astype('float')
        out = bfn(inp)
        exp_out = mdp.numx.zeros(lims[1][0] - lims[0][0] + 1)[None, :]
        exp_out[0, i] = 1.
        assert_array_equal(out, exp_out)
