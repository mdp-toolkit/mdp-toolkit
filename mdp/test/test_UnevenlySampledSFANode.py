from mdp.nodes.sfa_nodes import SFANode, UnevenlySampledSFANode
from mdp import numx, Node
from mdp.test._tools import assert_array_almost_equal, decimal


def test_UnevenlySampledSFANode1():
    # generate data
    x = numx.random.random((100000, 3))
    # minimal noise on time increment
    dt = (numx.random.rand(x.shape[0]-1)-.5)*0.01 + 1.

    # initialize nodes
    sfa = SFANode()
    unsfa = UnevenlySampledSFANode()

    # train sfa
    sfa.train(x)
    unsfa.train(x, dt)

    # stop training and generate slow features
    sfa.stop_training()
    unsfa.stop_training()

    if ((sfa.sf[0, 0] > 0) and (unsfa.sf[0, 0] < 0)) \
            or ((sfa.sf[0, 0] < 0) and (unsfa.sf[0, 0] > 0)):
        sfa.sf *= -1.

    assert_array_almost_equal(unsfa.sf, sfa.sf, decimal-6)
