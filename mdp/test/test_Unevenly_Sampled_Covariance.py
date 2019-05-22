"""
Tests for the UnevenlySampledCovarianceMatrix
"""
from mdp.utils.covariance import CovarianceMatrix,\
    UnevenlySampledCovarianceMatrix
from mdp import numx
from mdp.test._tools import assert_array_almost_equal, decimal


def test_UnevenlySampledCovarianceMatrix1():
    # initialize the estimators
    cov = CovarianceMatrix()
    uncov = UnevenlySampledCovarianceMatrix()
    # set sample distribution parameters
    tC = [[2., 0.5], [0.5, 1]]
    tAvg = [0, 0]
    # generate some data
    x = numx.random.multivariate_normal(tAvg, tC, 300000)
    # update the estimators
    cov.update(x)
    uncov.update(x, numx.ones((x.shape[0]-1,)))
    # quit estimating
    unC, unAvg, unTlen = uncov.fix()
    C, avg, tlen = cov.fix()

    # same for uneven, random time increments
    uncov = UnevenlySampledCovarianceMatrix()
    inc = (numx.random.rand(x.shape[0]-1)-.5)*.2 + 1.
    uncov.update(x, inc)
    unC2, unAvg2, unTlen2 = uncov.fix()

    assert_array_almost_equal(unC, tC, decimal-5)
    assert_array_almost_equal(unC2, tC, decimal-5)

    prec = numx.linalg.norm(tC-C)
    unPrec = numx.linalg.norm(tC-unC)
    unPrec2 = numx.linalg.norm(tC-unC2)

    assert_array_almost_equal(prec, unPrec, decimal-3)
    assert_array_almost_equal(prec, unPrec2, decimal-3)


def test_UnevenlySampledCovarianceMatrix2():
    # sample
    x = numx.random.random((10000, 2))
    dt = numx.ones(x.shape[0]-1,)
    # initialize the estimators
    cov = CovarianceMatrix()
    uncov = UnevenlySampledCovarianceMatrix()
    # update the estimators
    cov.update(x)
    uncov.update(x, dt)
    # quit estimating
    unC, unAvg, unTlen = uncov.fix()
    C, avg, tlen = cov.fix()

    assert_array_almost_equal(unC, C, decimal-3)
