"""These are test functions for MDP nodes.

Run them with:
>>> import mdp
>>> mdp.test("nodes")

"""
import unittest
import inspect
import mdp
import cPickle
import tempfile
import os
from mdp import utils, numx, numx_rand, numx_linalg, numx_fft
from testing_tools import assert_array_almost_equal, assert_array_equal, \
     assert_almost_equal, assert_equal, assert_array_almost_equal_diff, \
     assert_type_equal

mean = numx.mean
std = numx.std
normal = numx_rand.normal
uniform = numx_rand.random
testtypes = [numx.dtype('d'), numx.dtype('f')]
testtypeschar = [t.char for t in testtypes]
testdecimals = {testtypes[0]: 12, testtypes[1]: 6}

from _tools import (BogusNode, BogusNodeTrainable,
                    BogusExceptNode, BogusMultiNode)

def _rand_labels(x):
    return numx.around(uniform(x.shape[0]))

def _rand_labels_array(x):
    return numx.around(uniform(x.shape[0])).reshape((x.shape[0],1))

def _rand_array_halfdim(x):
    return uniform(size=(x.shape[0], x.shape[1]//2))

class NodesTestSuite(unittest.TestSuite):

    def __init__(self, testname=None):
        unittest.TestSuite.__init__(self)

        # constants
        self.mat_dim = (500,5)
        self.decimal = 7

        # set nodes to be tested
        self._set_nodes()

        if testname is not None:
            self._nodes_test_factory([testname])
        else:
            # get generic tests
            self._generic_test_factory()
            # get FastICA tests
            self._fastica_test_factory()
            # get nodes tests
            self._nodes_test_factory()

    def _set_nodes(self):
        mn = mdp.nodes
        self._nodes = [mn.PCANode,
                       mn.WhiteningNode,
                       mn.SFANode,
                       mn.SFA2Node,
                       mn.TDSEPNode,
                       mn.CuBICANode,
                       mn.FastICANode,
                       mn.QuadraticExpansionNode,
                       (mn.PolynomialExpansionNode, [3], None),
                       (mn.RBFExpansionNode, [[[0.]*5, [0.]*5], [1., 1.]], None),
                       mn.GrowingNeuralGasExpansionNode,
                       (mn.HitParadeNode, [2, 5], None),
                       (mn.TimeFramesNode, [3, 4], None),
                       mn.EtaComputerNode,
                       mn.GrowingNeuralGasNode,
                       mn.NoiseNode,
                       (mn.FDANode, [], _rand_labels),
                       (mn.GaussianClassifierNode, [], _rand_labels),
                       mn.FANode,
                       mn.ISFANode,
                       (mn.RBMNode, [5], None),
                       (mn.RBMWithLabelsNode, [5, 1], _rand_labels_array),
                       (mn.LinearRegressionNode, [], _rand_array_halfdim)]

