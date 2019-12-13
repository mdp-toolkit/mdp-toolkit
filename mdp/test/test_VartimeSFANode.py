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
    unsfa.train(x, dt=dt, time_dep=False)

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


def test_VartimeSFANode3():
    """Test whether different inputs for the same behavior result in the same
    output - without time dependence.
    """
    numx.random.seed(seed=10)
    # sample
    x1 = numx.random.random((1500, 2))
    x2 = numx.random.random((1500, 2))
    x3 = numx.random.random((1500, 2))
    xlen = x1.shape[0]
    dt_const = 1.
    dt_ones = numx.ones((xlen-1,))
    dt_none = None
    # initialize the nodes
    varsfa1 = VartimeSFANode()
    varsfa2 = VartimeSFANode()
    varsfa3 = VartimeSFANode()

    # update the estimators
    varsfa1.train(x1, dt=dt_const, time_dep=False)
    varsfa1.train(x2, dt=dt_const, time_dep=False)
    varsfa1.train(x3, dt=dt_const, time_dep=False)
    varsfa2.train(x1, dt=dt_ones, time_dep=False)
    varsfa2.train(x2, dt=dt_ones, time_dep=False)
    varsfa2.train(x3, dt=dt_ones, time_dep=False)
    varsfa3.train(x1, dt=dt_none, time_dep=False)
    varsfa3.train(x2, dt=dt_none, time_dep=False)
    varsfa3.train(x3, dt=dt_none, time_dep=False)
    # quit
    varsfa1.stop_training()
    varsfa2.stop_training()
    varsfa3.stop_training()

    # check whether it's the inverse
    if ((varsfa2.sf[0, 0] > 0) and (varsfa1.sf[0, 0] < 0)) \
            or ((varsfa2.sf[0, 0] < 0) and (varsfa1.sf[0, 0] > 0)):
        varsfa2.sf *= -1.
    if ((varsfa3.sf[0, 0] > 0) and (varsfa1.sf[0, 0] < 0)) \
            or ((varsfa3.sf[0, 0] < 0) and (varsfa1.sf[0, 0] > 0)):
        varsfa3.sf *= -1.

    assert_array_almost_equal(varsfa1.sf, varsfa2.sf, decimal=10)
    assert_array_almost_equal(varsfa1.sf, varsfa3.sf, decimal=10)


def test_VartimeSFANode4():
    """Test whether different inputs for the same behavior result in the same
    output - with time dependence.
    """
    numx.random.seed(seed=10)
    # sample
    x1 = numx.random.random((1500, 2))
    x2 = numx.random.random((1500, 2))
    x3 = numx.random.random((1500, 2))
    xlen = x1.shape[0]
    dt_const = 1.
    dt_ones = numx.ones((xlen,))
    dt_none = None
    # initialize the nodes
    varsfa1 = VartimeSFANode()
    varsfa2 = VartimeSFANode()
    varsfa3 = VartimeSFANode()

    # update the estimators
    varsfa1.train(x1, dt=dt_const, time_dep=True)
    varsfa2.train(x1, dt=dt_ones[1:], time_dep=True)
    varsfa3.train(x1, dt=dt_none, time_dep=True)

    varsfa1.train(x2, dt=dt_const, time_dep=True)
    varsfa2.train(x2, dt=dt_ones, time_dep=True)
    varsfa3.train(x2, dt=dt_none, time_dep=True)

    varsfa1.train(x3, dt=dt_const, time_dep=True)
    varsfa2.train(x3, dt=dt_ones, time_dep=True)
    varsfa3.train(x3, dt=dt_none, time_dep=True)
    # quit
    varsfa1.stop_training()
    varsfa2.stop_training()
    varsfa3.stop_training()

    # check whether it's the inverse
    if ((varsfa2.sf[0, 0] > 0) and (varsfa1.sf[0, 0] < 0)) \
            or ((varsfa2.sf[0, 0] < 0) and (varsfa1.sf[0, 0] > 0)):
        varsfa2.sf *= -1.
    if ((varsfa3.sf[0, 0] > 0) and (varsfa1.sf[0, 0] < 0)) \
            or ((varsfa3.sf[0, 0] < 0) and (varsfa1.sf[0, 0] > 0)):
        varsfa3.sf *= -1.

    assert_array_almost_equal(varsfa1.sf, varsfa2.sf, decimal=10)
    assert_array_almost_equal(varsfa1.sf, varsfa3.sf, decimal=10)
