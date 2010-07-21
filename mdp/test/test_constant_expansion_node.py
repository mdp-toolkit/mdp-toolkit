"""These are test functions for MDP constant expansion node.

Run them with:
>>> import mdp
>>> mdp.test("constant_expansion_node")

"""
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

##mult = utils.mult
##mean = numx.mean
##std = numx.std
##normal = numx_rand.normal
##uniform = numx_rand.random
##testtypes = [numx.dtype('d'), numx.dtype('f')]
##testtypeschar = [t.char for t in testtypes]
##testdecimals = {testtypes[0]: 12, testtypes[1]: 6}
##
##
##class _BogusNode(mdp.Node):
##    def is_trainable(self): return 0
##    def _execute(self,x): return 2*x
##    def _inverse(self,x): return 0.5*x
##
##class _BogusNodeTrainable(mdp.Node):
##    def _train(self, x):
##        pass
##    def _stop_training(self):
##        self.bogus_attr = 1
##
##class _BogusExceptNode(mdp.Node):
##    def _train(self,x):
##        self.bogus_attr = 1
##        raise Exception, "Bogus Exception"
##
##    def _execute(self,x):
##        raise Exception, "Bogus Exception"
##
##class _BogusMultiNode(mdp.Node):
##
##    def __init__(self):
##        super(_BogusMultiNode, self).__init__()
##        self.visited = []
##
##    def _get_train_seq(self):
##        return [(self.train1, self.stop1),
##                (self.train2, self.stop2)]
##
##    def train1(self, x):
##        self.visited.append(1)
##    def stop1(self):
##        self.visited.append(2)
##    def train2(self, x):
##        self.visited.append(3)
##    def stop2(self):
##        self.visited.append(4)
##
##
##def _rand_labels(x):
##    return numx.around(uniform(x.shape[0]))
##
##def _rand_labels_array(x):
##    return numx.around(uniform(x.shape[0])).reshape((x.shape[0],1))
##
##def _rand_array_halfdim(x):
##    return uniform(size=(x.shape[0], x.shape[1]//2))
##
##def _std(x):
##    return x.std(axis=0)
##    # standard deviation without bias
##    mx = mean(x, axis=0)
##    mx2 = mean(x*x, axis=0)
##    return numx.sqrt((mx2-mx)/(x.shape[0]-1))
##
##def _cov(x,y=None):
##    #return covariance matrix for x and y
##    if y is None:
##        y = x.copy()
##    x = x - mean(x,0)
##    x = x / _std(x)
##    y = y - mean(y,0)
##    y = y  / _std(y)
##    #return mult(numx.transpose(x),y)/(x.shape[0]-1)
##    return mult(numx.transpose(x),y)/(x.shape[0])
##
###_spinner = itertools.cycle((' /\b\b', ' -\b\b', ' \\\b\b', ' |\b\b'))
##_spinner = itertools.cycle((' .\b\b', ' o\b\b', ' 0\b\b', ' O\b\b',
##                            ' 0\b\b', ' o\b\b'))
###_spinner = itertools.cycle([" '\b\b"]*2 + [' !\b\b']*2 + [' .\b\b']*2 +
###                           [' !\b\b']*2)
##
### create spinner
##def spinner():
##    sys.stderr.write(_spinner.next())
##    sys.stderr.flush()

def dumb_quadratic_expansion(x):
    dim_x = x.shape[1]
    return numx.asarray([(x[i].reshape(dim_x,1) * x[i].reshape(1,dim_x)).flatten()
                       for i in range(len(x))])
    
class ConstantExpansionNodeTestSuite(unittest.TestCase):
    def testConstantExpansionNode(self):
        samples = 2
        input_dim = 10
        funcs = [lambda x:x, lambda x: x**2, dumb_quadratic_expansion]

        cen = mdp.nodes.ConstantExpansionNode(funcs)

        input = numx.random.normal(size=(samples, input_dim))         
        out = cen.execute(input)
        assert_array_almost_equal(out[:, 0:input_dim], input, 6, "incorrect constant expansion")
        assert_array_almost_equal(out[:, input_dim:2*input_dim], input ** 2, 6, "incorrect constant expansion")
        assert_array_almost_equal(out[:, 2*input_dim:], dumb_quadratic_expansion(input), 6, "incorrect constant expansion")
        assert cen.is_trainable() == False, "ConstantExpansionNode should be untrainable"
        assert cen.expanded_dim(input_dim) == 2 * input_dim + input_dim**2, "expanded_dim failed"
        assert_array_almost_equal(cen.output_sizes(input_dim), numx.array([input_dim, input_dim, input_dim*input_dim]), 6, "output_sizes failed")
    def testConstantExpansionNode_inverse(self):
        samples = 2
        input_dim = 10
        funcs = [lambda x:x, lambda x: x**2, dumb_quadratic_expansion]
        
        cen = mdp.nodes.ConstantExpansionNode(funcs, approximate_inverse=True, use_hint=True)
        input = numx.random.normal(size=(samples, input_dim))         
        out = cen.execute(input)
        app_input = cen.inverse(out)
        assert_array_almost_equal_diff(input, app_input, 6, 'inversion not good enough with use_hint=True')
        
        cen = mdp.nodes.ConstantExpansionNode(funcs, approximate_inverse=True, use_hint=False)
        input = numx.random.normal(size=(samples, input_dim))         
        out = cen.execute(input)
        app_input = cen.inverse(out)
        assert_array_almost_equal_diff(input, app_input, 4, 'inversion not good enough with use_hint=False')  

def get_suite(testname=None):
    return ConstantExpansionNodeTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.main()
