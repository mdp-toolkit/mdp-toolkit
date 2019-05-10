from mdp.nodes.sfa_nodes import SFANode, UnevenlySampledSFANode
from mdp import numx, Node
from mdp.test._tools import assert_array_almost_equal, decimal


def test_UnevenlySampledSFANode1():
    x = numx.random.random((10000,10))
    dt = (numx.random.rand(x.shape[0]-1)-.5)*.2 + 1.

    sfa = SFANode()
    unsfa = UnevenlySampledSFANode()

    sfa.train(x)
    unsfa.train(x, dt)

    sfa.stop_training()
    unsfa.stop_training()

    y = sfa.execute(x, n=2)
    uny = unsfa.execute(x, n=2)

    assert_array_almost_equal(y, uny, decimal-3)