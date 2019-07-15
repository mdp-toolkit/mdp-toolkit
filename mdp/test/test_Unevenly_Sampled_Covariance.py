"""
Tests for the UnevenlySampledCovarianceMatrix.
"""
from mdp.utils.covariance import CovarianceMatrix,\
    UnevenlySampledCovarianceMatrix
from mdp import numx
from mdp.test._tools import assert_array_almost_equal, decimal, assert_allclose


def test_UnevenlySampledCovarianceMatrix1():
    """Test if the new trazoidal rule integrator deviates substancially
    more than the regular one - with and without noisy input."""

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

    # precision of stepfunction covariance estimator
    prec = numx.linalg.norm(tC-C)
    # precision of trapezoidal convariacne matrix estimator
    # using non random step sizes
    unPrec = numx.linalg.norm(tC-unC)
    # precision of trapezoidal convariance matrix estimator
    # using random step sizes
    unPrec2 = numx.linalg.norm(tC-unC2)

    # allow deviation from standard by factor of .01 and .2 respectively
    assert_allclose(unPrec, prec, .01)
    assert_allclose(unPrec2, prec, .2)


def test_UnevenlySampledCovarianceMatrix2():
    """Test whether the trapezoidal integrator returns the expected
    based on the analytically adjusted results of the regular one."""

    # sample
    x = numx.random.random((10000, 2))
    dt = numx.ones(x.shape[0]-1,)
    # initialize the estimators
    cov = CovarianceMatrix(bias=True)
    uncov = UnevenlySampledCovarianceMatrix()
    # update the estimators
    cov.update(x)
    uncov.update(x, dt)
    # quit estimating
    unC, unAvg, unTlen = uncov.fix(center=False)
    C, avg, tlen = cov.fix(center=False)

    assert_array_almost_equal(unC*unTlen, C*tlen -
                              numx.outer(x[0], x[0])/2.-numx.outer(x[-1], x[-1])/2., decimal=10)
