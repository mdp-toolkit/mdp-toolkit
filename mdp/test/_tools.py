"""Tools for the test- and benchmark functions."""

import time
import mdp
numx = mdp.numx
import numpy.testing as testing
assert_array_equal, assert_array_almost_equal, \
     assert_equal, assert_almost_equal = \
     testing.assert_array_equal, testing.assert_array_almost_equal, \
     testing.assert_equal, testing.assert_almost_equal

mean = numx.mean
std = numx.std
normal = mdp.numx_rand.normal
uniform = mdp.numx_rand.random
testtypes = [numx.dtype('d'), numx.dtype('f')]
testtypeschar = [t.char for t in testtypes]
testdecimals = {testtypes[0]: 12, testtypes[1]: 6}


#### test tools
def assert_array_almost_equal_diff(x,y,digits,err_msg=''):
    x,y = numx.asarray(x), numx.asarray(y)
    msg = '\nArrays are not almost equal'
    assert 0 in [len(numx.shape(x)),len(numx.shape(y))] \
           or (len(numx.shape(x))==len(numx.shape(y)) and \
               numx.alltrue(numx.equal(numx.shape(x),numx.shape(y)))),\
               msg + ' (shapes %s, %s mismatch):\n\t' \
               % (numx.shape(x),numx.shape(y)) + err_msg
    maxdiff = max(numx.ravel(abs(x-y)))/\
              max(max(abs(numx.ravel(x))),max(abs(numx.ravel(y))))
    if numx.iscomplexobj(x) or numx.iscomplexobj(y): maxdiff = maxdiff/2
    cond =  maxdiff< 10**(-digits)
    msg = msg+'\n\t Relative maximum difference: %e'%(maxdiff)+'\n\t'+\
          'Array1: '+str(x)+'\n\t'+\
          'Array2: '+str(y)+'\n\t'+\
          'Absolute Difference: '+str(abs(y-x))
    assert cond, msg

def assert_type_equal(act, des):
    assert act == numx.dtype(des), \
           'dtype mismatch: "%s" (should be "%s") '%(act,des)

def get_random_mix(mat_dim = None, type = "d", scale = 1,\
                    rand_func = uniform, avg = 0, \
                    std_dev = 1):
    if mat_dim is None: mat_dim = (500, 5)
    T = mat_dim[0]
    N = mat_dim[1]
    d = 0
    while d < 1E-3:
        #mat = ((rand_func(size=mat_dim)-0.5)*scale).astype(type)
        mat = rand_func(size=(T,N)).astype(type)
        # normalize
        mat -= mean(mat,axis=0)
        mat /= std(mat,axis=0)
        # check that the minimum eigenvalue is finite and positive
        d1 = min(mdp.utils.symeig(mdp.utils.mult(mat.T, mat), eigenvectors = 0))
        if std_dev is not None: mat *= std_dev
        if avg is not None: mat += avg
        mix = (rand_func(size=(N,N))*scale).astype(type)
        matmix = mdp.utils.mult(mat,mix)
        matmix_n = matmix - mean(matmix, axis=0)
        matmix_n /= std(matmix_n, axis=0)
        d2 = min(mdp.utils.symeig(mdp.utils.mult(matmix_n.T,matmix_n),eigenvectors=0))
        d = min(d1, d2)
    return mat, mix, matmix
