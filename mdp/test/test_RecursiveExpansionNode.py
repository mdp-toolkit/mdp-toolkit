"""
Tests for the RecursiveExpansionNode.
"""

from mdp.nodes import PolynomialExpansionNode
from mdp.nodes import (RecursiveExpansionNode,
                       NormalizingRecursiveExpansionNode)
from mdp.nodes.recursive_expansion_nodes import recfs
from mdp.test._tools import *
from mdp import numx as np
import py.test


def test_RecursiveExpansionNode1():
    """Testing the one-dimensional expansion."""
    for functup in funcs:
        func = functup[0]
        degree = functup[1]
        name = functup[2]
        data = np.random.rand(1, 1)

        expn = RecursiveExpansionNode(
            degree, recf=name, check=False, with0=True)
        nodeexp = expn.execute(data)
        assert_array_almost_equal(nodeexp,
                                  func(data[:, 0], degree), decimal-3)
        print('Single dim '+name + ' equal')


def test_RecursiveExpansionNode2():
    """Testing the tensor-base."""
    data = 1e-6+np.random.rand(10, 4)
    for functup in funcs:
        func = functup[0]
        degree = 4
        name = functup[2]
        recexpn = RecursiveExpansionNode(degree, recf=name,
                                         check=False, with0=True)
        resrec = recexpn.execute(data)
        restensor = np.array([get_handcomputed_function_tensor(data[i, :], func, degree)
                              for i in range(data.shape[0])])

        ressimplex = np.array(
            [makeSimplex(restensor[i, :, :, :, :]) for i in range(data.shape[0])])
        assert_array_almost_equal(resrec, ressimplex, decimal-3)
        print('Multi dim ' + name + ' equal')


def get_chebyshev_poly(x, degree):
    p = np.ndarray((x.shape[0], 7))
    p[:, 0] = 1.
    p[:, 1] = x
    p[:, 2] = 2.*x*x-1.
    p[:, 3] = 4.*x**3-3.*x
    p[:, 4] = 8.*x**4-8.*x**2+1.
    p[:, 5] = 16.*x**5-20.*x**3+5.*x
    p[:, 6] = 32.*x**6-48.*x**4+18.*x*x-1.
    return p[:, :degree+1]


def get_legendre_ratio(x, degree):
    p = np.ndarray((x.shape[0], 5))
    p[:, 0] = 1.
    p[:, 1] = (x-1.)/(x+1.)
    p[:, 2] = (x*x-4.*x+1.)/(x+1.)**2
    p[:, 3] = (x**3-9.*x*x+9.*x-1.)/(x+1.)**3
    p[:, 4] = (x**4-16.*x**3+36.*x*x-16.*x+1.)/(x+1.)**4
    return p[:, :degree+1]


def get_legendre_poly(x, degree):
    p = np.ndarray((x.shape[0], 7))
    p[:, 0] = 1.
    p[:, 1] = x
    p[:, 2] = 0.5*(3.*x*x-1.)
    p[:, 3] = 0.5*(5.*x*x-3.)*x
    p[:, 4] = 1./8.*(35.*x**4-30.*x*x+3.)
    p[:, 5] = 1./8.*(63.*x**5-70.*x**3+15.*x)
    p[:, 6] = 1./16.*(231.*x**6-315.*x**4+105.*x*x-5.)
    return p[:, :degree+1]


def get_standard_poly(x, degree):
    p = np.ndarray((x.shape[0], 7))
    p[:, 0] = 1.
    p[:, 1] = x
    p[:, 2] = x*x
    p[:, 3] = x**3
    p[:, 4] = x**4
    p[:, 5] = x**5
    p[:, 6] = x**6
    return p[:, :degree+1]


def get_handcomputed_function_tensor(x, func, degree):
    """x must be of shape (4,)."""
    outtensor = np.zeros((degree+1,)*4)

    outtensor[:, 0, 0, 0] = func(x[np.newaxis, 0], degree)
    outtensor[0, :, 0, 0] = func(x[np.newaxis, 1], degree)
    outtensor[0, 0, :, 0] = func(x[np.newaxis, 2], degree)
    outtensor[0, 0, 0, :] = func(x[np.newaxis, 3], degree)
    for i in range(degree+1):
        outtensor[:, i, 0, 0] = outtensor[:, 0, 0, 0]*outtensor[0, i, 0, 0]

    for i in range(degree+1):
        outtensor[:, :, i, 0] = outtensor[:, :, 0, 0] * outtensor[0, 0, i, 0]

    for i in range(degree+1):
        outtensor[:, :, :, i] = outtensor[:, :, :, 0] * outtensor[0, 0, 0, i]

    return outtensor


funcs = [(get_legendre_poly, 6, 'legendre_poly'),
         (get_legendre_ratio, 4, 'legendre_rational'),
         (get_standard_poly, 6, 'standard_poly'),
         (get_chebyshev_poly, 6, 'chebyshev_poly')]


def makeSimplex(tensor):
    simplex = np.concatenate(
        (tensor[:, 0, 0, 0], tensor[0, 1:, 0, 0], tensor[0, 0, 1:, 0],
         tensor[0, 0, 0, 1:]))
    x1 = tensor[1, 0, 0, 0]
    x2 = tensor[2, 0, 0, 0]
    x3 = tensor[3, 0, 0, 0]

    y1 = tensor[0, 1, 0, 0]
    y2 = tensor[0, 2, 0, 0]
    y3 = tensor[0, 3, 0, 0]

    z1 = tensor[0, 0, 1, 0]
    z2 = tensor[0, 0, 2, 0]
    z3 = tensor[0, 0, 3, 0]

    w1 = tensor[0, 0, 0, 1]
    w2 = tensor[0, 0, 0, 2]
    w3 = tensor[0, 0, 0, 3]

    simplex = np.concatenate((simplex, x1 * np.array([y1, y2, y3])))
    simplex = np.concatenate((simplex, x2 * np.array([y1, y2])))
    simplex = np.concatenate((simplex, np.array([x3 * y1])))

    simplex = np.concatenate((simplex, x1 * np.array([z1, z2, z3])))
    simplex = np.concatenate((simplex, x2 * np.array([z1, z2])))
    simplex = np.concatenate((simplex, x3 * np.array([z1])))

    simplex = np.concatenate((simplex, y1 * np.array([z1, z2, z3])))
    simplex = np.concatenate((simplex, y2 * np.array([z1, z2])))
    simplex = np.concatenate((simplex, y3 * np.array([z1])))

    simplex = np.concatenate((simplex, x1 * np.array([w1, w2, w3])))
    simplex = np.concatenate((simplex, x2 * np.array([w1, w2])))
    simplex = np.concatenate((simplex, x3 * np.array([w1])))

    simplex = np.concatenate((simplex, y1 * np.array([w1, w2, w3])))
    simplex = np.concatenate((simplex, y2 * np.array([w1, w2])))
    simplex = np.concatenate((simplex, y3 * np.array([w1])))

    simplex = np.concatenate((simplex, z1 * np.array([w1, w2, w3])))
    simplex = np.concatenate((simplex, z2 * np.array([w1, w2])))
    simplex = np.concatenate((simplex, z3 * np.array([w1])))

    simplex = np.concatenate((simplex, x1 * np.array([y1*z1, y1*z2])))
    simplex = np.concatenate((simplex, x2 * np.array([y1*z1])))
    simplex = np.concatenate((simplex, x1 * np.array([y2*z1])))

    simplex = np.concatenate((simplex, x1 * np.array([y1*w1, y1*w2])))
    simplex = np.concatenate((simplex, x2 * np.array([y1*w1])))
    simplex = np.concatenate((simplex, x1 * np.array([y2*w1])))

    simplex = np.concatenate((simplex, x1 * np.array([z1*w1, z1*w2])))
    simplex = np.concatenate((simplex, x2 * np.array([z1*w1])))
    simplex = np.concatenate((simplex, y1 * np.array([z1*w1, z1*w2])))

    simplex = np.concatenate((simplex, y2 * np.array([z1*w1])))
    simplex = np.concatenate((simplex, x1 * np.array([z2*w1])))
    simplex = np.concatenate((simplex, y1 * np.array([z2*w1])))

    simplex = np.concatenate((simplex, x1 * np.array([y1*z1*w1])))

    return simplex
