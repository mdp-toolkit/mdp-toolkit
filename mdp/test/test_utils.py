"""These are test functions for MDP utilities.

Run them with:
>>> import mdp
>>> mdp.test("utils")

"""
import unittest
import pickle
import os
import tempfile
import platform
import inspect
from mdp import numx, utils, numx_rand, numx_linalg, Node, nodes, MDPException
from testing_tools import assert_array_almost_equal, assert_array_equal, \
     assert_almost_equal, assert_equal, assert_array_almost_equal_diff, \
     assert_type_equal

testtypes = [numx.dtype('d'),numx.dtype('f'),numx.dtype('D'),numx.dtype('F')]
testdecimals = {testtypes[0]: 12, testtypes[1]: 3,
                testtypes[2]: 12, testtypes[3]: 3}

class BogusClass(object):
    def __init__(self):
        self.x = numx_rand.random((2,2))
    
class BogusNode(Node):
    x = numx_rand.random((2,2))
    y = BogusClass()
    z = BogusClass()
    z.z = BogusClass()

class UtilsTestSuite(unittest.TestSuite):
##     def testProgressBar(self):
##         print
##         p = utils.ProgressBar(minimum=0,maximum=1000)
##         for i in range(1000):
##             p.update(i+1)
##             for j in xrange(10000): pass
##         print

    def __init__(self, testname=None):
        unittest.TestSuite.__init__(self)

        if testname is not None:
            self._utils_test_factory([testname])
        else:
            # get all tests
            self._utils_test_factory()

    def _utils_test_factory(self, methods_list=None):
        if methods_list is None:
            methods_list = dir(self)
        for methname in methods_list:
            try:
                meth = getattr(self,methname)
            except AttributeError:
                continue
            if inspect.ismethod(meth) and meth.__name__[:4] == "test":
                # create a nice description
                descr = 'Test '+(meth.__name__[4:]).replace('_',' ')
                self.addTest(unittest.FunctionTestCase(meth,
                             description=descr))

    def eigenproblem(self, dtype, range, func=utils._symeig_fake):
        """Solve a standard eigenvalue problem."""
        dtype = numx.dtype(dtype)
        dim = 5
        if range:
            range = (2, dim -1)
        else:
            range = None
        a = utils.symrand(dim, dtype)
        w,z = func(a, range=range)
        # assertions
        assert_type_equal(z.dtype, dtype)
        w = w.astype(dtype)
        diag = numx.diagonal(utils.mult(utils.hermitian(z),
                                        utils.mult(a, z))).real
        assert_array_almost_equal(diag, w, testdecimals[dtype])

    def geneigenproblem(self, dtype, range):
        """Solve a generalized eigenvalue problem."""
        """Solve a standard eigenvalue problem."""
        dtype = numx.dtype(dtype)
        dim = 5
        if range:
            range = (2, dim -1)
        else:
            range = None
        a = utils.symrand(dim, dtype)
        b = utils.symrand(dim, dtype)
        w,z = utils._symeig_fake(a,b,range=range)
        # assertions
        assert_type_equal(z.dtype, dtype)
        w = w.astype(dtype)
        diag1 = numx.diagonal(utils.mult(utils.hermitian(z),
                                         utils.mult(a, z))).real
        assert_array_almost_equal(diag1, w, testdecimals[dtype])
        diag2 = numx.diagonal(utils.mult(utils.hermitian(z),
                                         utils.mult(b, z))).real
        assert_array_almost_equal(diag2, numx.ones(diag2.shape[0]),
                                  testdecimals[dtype] )


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
            y = utils.mult(x.T, x)
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

    def testMultDiag(self):
        dim = 20
        d = numx_rand.random(size=(dim,))
        dd = numx.diag(d)
        mtx = numx_rand.random(size=(dim, dim))
        
        res1 = utils.mult(dd, mtx)
        res2 = utils.mult_diag(d, mtx, left=True)
        assert_array_almost_equal(res1, res2, 10)
        res1 = utils.mult(mtx, dd)
        res2 = utils.mult_diag(d, mtx, left=False)
        assert_array_almost_equal(res1, res2, 10)

    def testSymeig_fake_standard(self):
        self.eigenproblem('d',False)
        self.eigenproblem('f',False)
        self.eigenproblem('d',True)
        self.eigenproblem('f',True)

    def testSVD_standard(self):
        func = utils.nongeneral_svd
        self.eigenproblem('d',False, func=func)
        self.eigenproblem('f',False, func=func)
        self.eigenproblem('d',True, func=func)
        self.eigenproblem('f',True, func=func)

    def testSymeig_fake_general(self):
        self.geneigenproblem('d',False)
        self.geneigenproblem('f',False)
        self.geneigenproblem('d',True)
        self.geneigenproblem('f',True)

    def testSymeig_fake_integer(self):
        a = numx.array([[1,2],[2,7]])
        b = numx.array([[3,1],[1,5]])
        w,z = utils._symeig_fake(a)
        w,z = utils._symeig_fake(a,b)

    def testSymeig_fake_LAPACK_bug(self):
        # bug. when input matrix is almost an identity matrix
        # but not exactly, the lapack dgeev routine returns a
        # matrix of eigenvectors which is not orthogonal.
        # this bug was present when we used numx_linalg.eig
        # instead of numx_linalg.eigh .
        # Note: this is a LAPACK bug.
        y = numx_rand.random((4,4))*1E-16
        y = (y+y.T)/2
        for i in range(4):
            y[i,i]=1
        val, vec = utils._symeig_fake(y)
        assert_almost_equal(abs(numx_linalg.det(vec)), 1., 12)



    def testQuadraticFormsExtrema(self):
        # !!!!! add some real test
        # check H with negligible linear term
        noise = 1e-8
        tol = 1e-6
        x = numx_rand.random((10,))
        H = numx.outer(x, x) + numx.eye(10)*0.1
        f = noise*numx_rand.random((10,))
        q = utils.QuadraticForm(H, f)
        xmax, xmin = q.get_extrema(utils.norm2(x), tol=tol)
        assert_array_almost_equal(x, xmax, 5)
        # check I + linear term
        H = numx.eye(10, dtype='d')
        f = x
        q = utils.QuadraticForm(H, f=f)
        xmax, xmin = q.get_extrema(utils.norm2(x), tol=tol) 
        assert_array_almost_equal(f, xmax, 5)

    def testQuadraticFormsInvariances(self):
        # don't run test on 64bit platform because
        # of bug in linalg.qr (already fixed in numpy svn)
        if platform.architecture()[0] != '32bit':
            return
        #nu = numx.linspace(2.,-3,10)
        nu = numx.linspace(6., 1, 10)
        H = utils.symrand(nu)
        E, W = utils.symeig(H)
        q = utils.QuadraticForm(H)
        xmax, xmin = q.get_extrema(5.)
        e_w, e_sd = q.get_invariances(xmax)
        assert_array_almost_equal(e_sd,nu[1:]-nu[0],6)
        assert_array_almost_equal(abs(e_w),abs(W[:,-2::-1]),6)
        e_w, e_sd = q.get_invariances(xmin)
        assert_array_almost_equal(e_sd,nu[-2::-1]-nu[-1],6)
        assert_array_almost_equal(abs(e_w),abs(W[:,1:]),6)

    def testQuadraticFormsException(self):
        H = numx_rand.random((10,10))
        try:
            q = utils.QuadraticForm(H)
        except MDPException, e:
            if 'H does not seem to be symmetric' in str(e):
                return
            else:
                raise e
        raise Exception, 'Did not detect non symmetric H!'
    
def get_suite(testname=None):
    return UtilsTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
