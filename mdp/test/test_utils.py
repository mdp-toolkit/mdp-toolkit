"""These are test functions for MDP utilities.

Run them with:
>>> import mdp
>>> mdp.test.test("utils")

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

    def testCrashRecoveryException(self):
        a = 3
        try:
            raise utils.CrashRecoveryException, \
                  ('bogus errstr',a,StandardError())
        except utils.CrashRecoveryException, e:
            filename1 = e.dump()
            filename2 = e.dump(os.path.join(tempfile.gettempdir(),'removeme'))
            assert isinstance(e.parent_exception, StandardError)

        for fname in [filename1,filename2]:
            fl = file(fname)
            obj = pickle.load(fl)
            fl.close()
            os.remove(fname)
            assert obj == a
            
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
                
def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(UtilsTestCase))
    return suite

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
