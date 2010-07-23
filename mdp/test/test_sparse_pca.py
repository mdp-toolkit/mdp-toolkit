# -*- coding: utf-8 -*-
#--------------------------
# boilerplate, copied from test_nodes
#--------------------------
import sys
sys.path.append("/home/quesada/coding/projIfollow/mdp-toolkit/")
import unittest
import inspect
import mdp
import cPickle
import tempfile
import os
import itertools
import sys
from mdp import utils, numx, numx_rand, numx_linalg, numx_fft
from testing_tools import assert_array_almost_equal, assert_array_equal, \
assert_almost_equal, assert_equal, assert_array_almost_equal_diff, \
assert_type_equal

print numx
from scipy import sparse
from scipy.sparse import linalg
print numx.__version__


mult = utils.mult
mean = numx.mean
std = numx.std
normal = numx_rand.normal
uniform = numx_rand.random
testtypes = [numx.dtype('d'), numx.dtype('f')]
testtypeschar = [t.char for t in testtypes]
testdecimals = {testtypes[0]: 12, testtypes[1]: 6}

#--------------------------
# simply pass a sparse matrix
#--------------------------
#import scipy
#from numpy.random import rand

# create pca node
pcanode1 = mdp.nodes.SparsePCANode()

 
# create random sparse matrix
x = mdp.numx.sparse.lil_matrix((1000, 1000))
x[0, :100] = mdp.numx.rand(100)
x[1, 100:200] = x[0, :100]
x.setdiag(mdp.numx.rand(1000))
x = x.tocsr() # convert it to CSR

v, u, w = numx.sparse.linalg.eigen_symmetric(x)

pcanode1.train(x)
pcanode1.stop_training()

