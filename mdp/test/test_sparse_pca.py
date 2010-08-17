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

x = mdp.numx.sparse.lil_matrix((12, 11))
for i in range(11):
    #x[i,i] = float(i)
    #x[i,i] = float(i)
    x[0,i] = float(i)+1.
    x[i+1,i] = -float(i)-1.
    
#d, v = numx.sparse.linalg.eigen_symmetric(x, k=10)




 
# create random sparse matrix
#x = mdp.numx.sparse.lil_matrix((1000, 1000))
#[0, :100] = mdp.numx.rand(100)
#x[1, 100:200] = x[0, :100]
#x.setdiag(mdp.numx.rand(1000))
#x = x.tocsr() # convert it to CSR


# create pca node
pcanode1 = mdp.nodes.SparsePCANode(output_dim=10)

pcanode1.train(x)
pcanode1.stop_training(debug=True)

#print pcanode1.d, d
#assert all(pcanode1.d == d)