"""These are test functions for MDP utilities.

Run them with:
>>> import mdp
>>> mdp.test("utils")

"""
import unittest
import pickle
import os
import tempfile
from mdp import numx, utils, numx_rand, numx_linalg, Node, nodes
from testing_tools import assert_array_almost_equal, assert_array_equal, \
     assert_almost_equal, assert_equal, assert_array_almost_equal_diff, \
     assert_type_equal

class BogusClass(object):
    def __init__(self):
        self.x = numx_rand.random((2,2))
    
class BogusNode(Node):
    x = numx_rand.random((2,2))
    y = BogusClass()
    z = BogusClass()
    z.z = BogusClass()

class UtilsTestCase(unittest.TestCase):
##     def testProgressBar(self):
##         print
##         p = utils.ProgressBar(minimum=0,maximum=1000)
##         for i in range(1000):
##             p.update(i+1)
##             for j in xrange(10000): pass
##         print
            
    def testIntrospection(self):
        bogus = BogusNode()
        arrays, string = utils.dig_node(bogus)
        assert len(arrays.keys()) == 4, 'Not all arrays where caught'
        assert sorted(arrays.keys()) == ['x', 'y.x',
                                         'z.x', 'z.z.x'], 'Wrong names'
        sizes = [x[0] for x in arrays.values()]
        assert sorted(sizes) == [numx_rand.random((2,2)).itemsize*4]*4, \
               'Wrong sizes'
        sfa = nodes.SFANode()
        sfa.train(numx_rand.random((1000, 10)))
        a_sfa, string = utils.dig_node(sfa)
        keys = ['_cov_mtx._avg', '_cov_mtx._cov_mtx',
                '_dcov_mtx._avg', '_dcov_mtx._cov_mtx',]
        assert sorted(a_sfa.keys()) == keys, 'Wrong arrays in SFANode'
        sfa.stop_training()
        a_sfa, string = utils.dig_node(sfa)
        keys = ['avg', 'd', 'sf']
        assert sorted(a_sfa.keys()) == keys, 'Wrong arrays in SFANode'

    def testRandomRot(self):
        dim = 20
        tlen = 10
        for i in range(tlen):
            x = utils.random_rot(dim, dtype='f')
            assert x.dtype.char=='f', 'Wrong dtype'
            y = utils.mult(numx.transpose(x), x)
            assert_almost_equal(numx_linalg.det(x), 1., 4)
            assert_array_almost_equal(y, numx.eye(dim), 4)

    def testCasting(self):
        x = numx_rand.random((5,3)).astype('d')
        y = 3*x
        assert_type_equal(y.dtype, x.dtype)
        x = numx_rand.random((5,3)).astype('f')
        y = 3.*x
        assert_type_equal(y.dtype, x.dtype)
        x = (10*numx_rand.random((5,3))).astype('i')
        y = 3.*x
        assert_type_equal(y.dtype, 'd')
        y = 3L*x
        assert_type_equal(y.dtype, 'i')
        x = numx_rand.random((5,3)).astype('f')
        y = 3L*x
        assert_type_equal(y.dtype, 'f')

    def testQuadraticForms(self):
        # !!!!! add some real test
        # check H with negligible linear term
        noise = 1e-7
        x = numx_rand.random((10,))
        H = numx.outer(x, x) + numx.eye(10)*0.1
        f = noise*numx_rand.random((10,))
        q = utils.QuadraticForm(H=H, f=f, c=0.)
        xmax, xmin, vmax, vmin = q.get_extrema(utils.norm2(x), tol=noise)
        assert_array_almost_equal(x, xmax, 5)
        # check I + linear term
        H = numx.eye(10, dtype='d')
        f = x
        q = utils.QuadraticForm(H=H, f=f, c=0.)
        xmax, xmin, vmax, vmin = q.get_extrema(utils.norm2(x), tol=noise) 
        assert_array_almost_equal(f, xmax, 5)
                 
def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(UtilsTestCase))
    return suite

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
