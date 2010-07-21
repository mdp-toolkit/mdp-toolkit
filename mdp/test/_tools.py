"""Tools for the test- and benchmark functions."""

import time
import mdp
numx = mdp.numx
import numpy.testing as testing
assert_array_equal, assert_array_almost_equal, \
     assert_equal, assert_almost_equal = \
     testing.assert_array_equal, testing.assert_array_almost_equal, \
     testing.assert_equal, testing.assert_almost_equal

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
