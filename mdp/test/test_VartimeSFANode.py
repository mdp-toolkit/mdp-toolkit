"""
Tests for the VartimeSFANode.
"""
from mdp.nodes.sfa_nodes import SFANode, VartimeSFANode
from mdp import numx, Node
from mdp.test._tools import assert_array_almost_equal, decimal


def test_VartimeSFANode1():
    """Check whether solutions are rougly close to original solutions of sfa."""
    # generate data
    x = numx.random.random((100000, 3))
    # minimal noise on time increment
    dt = (numx.random.rand(x.shape[0]-1)-.5)*0.01 + 1.

    # initialize nodes
    sfa = SFANode()
    unsfa = VartimeSFANode()

    # train sfa
    sfa.train(x)
    unsfa.train(x, dt=dt)

    # stop training and generate slow features
    sfa.stop_training()
    unsfa.stop_training()

    # check whether it's the inverse
    if ((sfa.sf[0, 0] > 0) and (unsfa.sf[0, 0] < 0)) \
            or ((sfa.sf[0, 0] < 0) and (unsfa.sf[0, 0] > 0)):
        sfa.sf *= -1.

    assert_array_almost_equal(unsfa.sf, sfa.sf, decimal-6)


def test_VartimeSFANode2():
    """Check whether splitting input to unsfa in multiple chunks 
    and with time dependence works.
    """
    x = numx.random.random((12000, 2))
    dt = (numx.random.rand(x.shape[0]-1)-.5)*.5 + 1.
    xlen = x.shape[0]
    # initialize the nodes
    unsfa = VartimeSFANode()
    unsfa2 = VartimeSFANode()

    # update the estimators
    unsfa2.train(x, dt=dt)

    # split into phases
    dtpart1 = dt[:xlen//3-1]
    dtpart2 = dt[xlen//3-1:2*xlen//3-1]
    dtpart3 = dt[2*xlen//3-1:]
    xpart1 = x[:xlen//3]
    xpart2 = x[xlen//3:2*xlen//3]
    xpart3 = x[2*xlen//3:]

    # train
    unsfa.train(xpart1, dt=dtpart1)
    unsfa.train(xpart2, dt=dtpart2)
    unsfa.train(xpart3, dt=dtpart3)

    # quit
    unsfa.stop_training()
    unsfa2.stop_training()

    # check whether it's the inverse
    if ((unsfa2.sf[0, 0] > 0) and (unsfa.sf[0, 0] < 0)) \
            or ((unsfa2.sf[0, 0] < 0) and (unsfa.sf[0, 0] > 0)):
        unsfa2.sf *= -1.

    assert_array_almost_equal(unsfa.sf, unsfa2.sf, decimal=10)
