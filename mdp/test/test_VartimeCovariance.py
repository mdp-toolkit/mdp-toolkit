"""
Tests for the VartimeCovarianceMatrix.
"""
from mdp.utils.covariance import CovarianceMatrix,\
    VartimeCovarianceMatrix
from mdp import numx
from mdp.test._tools import assert_array_almost_equal, decimal, assert_allclose


def test_VartimeCovarianceMatrix1():
    """Test if the new trapezoidal rule integrator deviates substantially
    more than the regular one - with and without noisy input.
    """

    # initialize the estimators
    cov = CovarianceMatrix()
    uncov = VartimeCovarianceMatrix()
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
    uncov = VartimeCovarianceMatrix()
    inc = (numx.random.rand(x.shape[0]-1)-.5)*.2 + 1.
    uncov.update(x, inc)
    unC2, unAvg2, unTlen2 = uncov.fix()

    # precision of step function covariance estimator
    prec = numx.linalg.norm(tC-C)
    # precision of trapezoidal covariance matrix estimator
    # using non random step sizes
    unPrec = numx.linalg.norm(tC-unC)
    # precision of trapezoidal covariance matrix estimator
    # using random step sizes
    unPrec2 = numx.linalg.norm(tC-unC2)

    # allow deviation from standard by factor of .01 and .2 respectively
    assert_allclose(unPrec, prec, .01)
    assert_allclose(unPrec2, prec, .2)


def test_VartimeCovarianceMatrix2():
    """Test whether the trapezoidal integrator returns the expected
    based on the analytically adjusted results of the regular one.
    """

    # sample
    x = numx.random.random((10000, 2))
    dt = numx.ones(x.shape[0]-1,)
    # initialize the estimators
    cov = CovarianceMatrix(bias=True)
    uncov = VartimeCovarianceMatrix()
    # update the estimators
    cov.update(x)
    uncov.update(x, dt)
    # quit estimating
    unC, unAvg, unTlen = uncov.fix(center=False)
    C, avg, tlen = cov.fix(center=False)

    assert_array_almost_equal(unC*unTlen, C*tlen -
                              numx.outer(x[0], x[0])/2.-numx.outer(x[-1], x[-1])/2., decimal=10)


def test_VartimeCovarianceMatrix3():
    """Test whether the trapezoidal integrator returns the expected
    when calculated in multiple phases and without time dependence.
    """

    # sample
    x = numx.random.random((15000, 2))
    dt = (numx.random.rand(x.shape[0]-1)-.5)*.5 + 1.
    xlen = x.shape[0]
    # initialize the estimators
    uncov = VartimeCovarianceMatrix()
    uncov2 = VartimeCovarianceMatrix()
    # emulate disconnection
    dt[xlen//3-1] = 0.
    dt[2*xlen//3-1] = 0.
    # update the estimators
    uncov.update(x, dt)

    # split into phases
    dtpart1 = dt[:xlen//3-1]
    dtpart2 = dt[xlen//3:2*xlen//3-1]
    dtpart3 = dt[2*xlen//3:]
    xpart1 = x[:xlen//3]
    xpart2 = x[xlen//3:2*xlen//3]
    xpart3 = x[2*xlen//3:]

    uncov2.update(xpart1, dtpart1)
    uncov2.update(xpart2, dtpart2)
    uncov2.update(xpart3, dtpart3)
    # quit estimating
    unC, unAvg, unTlen = uncov.fix(center=True)
    unC2, unAvg2, unTlen2 = uncov2.fix(center=True)

    assert_array_almost_equal(unC, unC2, decimal=10)


def test_VartimeCovarianceMatrix4():
    """Test whether the trapezoidal integrator returns the expected
    when calculated in multiple phases and time dependence.
    """

    # sample
    x = numx.random.random((15000, 2))
    dt = (numx.random.rand(x.shape[0]-1)-.5)*.5 + 1.
    xlen = x.shape[0]
    # initialize the estimators
    uncov = VartimeCovarianceMatrix()
    uncov2 = VartimeCovarianceMatrix()
    # update the estimators
    uncov.update(x, dt)

    # split into phases
    dtpart1 = dt[:xlen//3-1]
    dtpart2 = dt[xlen//3-1:2*xlen//3-1]
    dtpart3 = dt[2*xlen//3-1:]
    xpart1 = x[:xlen//3]
    xpart2 = x[xlen//3:2*xlen//3]
    xpart3 = x[2*xlen//3:]

    uncov2.update(xpart1, dtpart1)
    uncov2.update(xpart2, dtpart2)
    uncov2.update(xpart3, dtpart3)
    # quit estimating
    unC, unAvg, unTlen = uncov.fix(center=False)
    unC2, unAvg2, unTlen2 = uncov2.fix(center=False)

    assert_array_almost_equal(unC, unC2, decimal=10)
