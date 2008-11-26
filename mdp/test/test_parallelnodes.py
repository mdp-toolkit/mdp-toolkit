
import unittest

import mdp
import mdp.parallel as parallel
from mdp import utils, numx, numx_rand
from testing_tools import assert_array_almost_equal, assert_almost_equal


class TestParallelMDPNodes(unittest.TestCase):
    
    def test_PCANode(self):
        """Test Parallel PCANode"""
        precision = 6
        x = numx_rand.random([100,10])
        x_test = numx_rand.random([20,10])
        # set different variances (avoid numerical errors)
        x *= numx.arange(1,11)
        x_test *= numx.arange(1,11)
        pca_node = mdp.nodes.PCANode()
        parallel_pca_node = parallel.ParallelPCANode()
        chunksize = 25
        chunks = [x[i*chunksize : (i+1)*chunksize] 
                    for i in range(len(x)/chunksize)]
        for chunk in chunks:
            pca_node.train(chunk)
            forked_node = parallel_pca_node.fork()
            forked_node.train(chunk)
            parallel_pca_node.join(forked_node)
        assert_array_almost_equal(pca_node._cov_mtx._cov_mtx, 
                                  parallel_pca_node._cov_mtx._cov_mtx, 
                                  precision)
        pca_node.stop_training()
        y1 = pca_node.execute(x_test)
        parallel_pca_node.stop_training()
        y2 = parallel_pca_node.execute(x_test)
        assert_array_almost_equal(abs(y1), abs(y2), precision)
        
    def test_WhiteningNode(self):
        """Test Parallel WhiteningNode"""
        precision = 6
        x = numx_rand.random([100,10])
        x_test = numx_rand.random([20,10])
        # set different variances (avoid numerical errors)
        x *= numx.arange(1,11)
        x_test *= numx.arange(1,11)
        pca_node = mdp.nodes.WhiteningNode()
        parallel_pca_node = parallel.ParallelWhiteningNode()
        chunksize = 25
        chunks = [x[i*chunksize : (i+1)*chunksize] 
                    for i in range(len(x)/chunksize)]
        for chunk in chunks:
            pca_node.train(chunk)
            forked_node = parallel_pca_node.fork()
            forked_node.train(chunk)
            parallel_pca_node.join(forked_node)
        assert_array_almost_equal(pca_node._cov_mtx._cov_mtx, 
                                  parallel_pca_node._cov_mtx._cov_mtx, 
                                  precision)
        pca_node.stop_training()
        y1 = pca_node.execute(x_test)
        parallel_pca_node.stop_training()
        y2 = parallel_pca_node.execute(x_test)
        assert_array_almost_equal(abs(y1), abs(y2), precision)

    def test_SFANode(self):
        """Test Parallel SFANode"""
        precision = 6
        x = numx_rand.random([100,10])
        x_test = numx_rand.random([20,10])
        # set different variances (avoid numerical errors)
        x *= numx.arange(1,11)
        x_test *= numx.arange(1,11)
        sfa_node = mdp.nodes.SFANode()
        parallel_sfa_node = parallel.ParallelSFANode()
        chunksize = 25
        chunks = [x[i*chunksize : (i+1)*chunksize] 
                    for i in range(len(x)/chunksize)]
        for chunk in chunks:
            sfa_node.train(chunk)
            forked_node = parallel_sfa_node.fork()
            forked_node.train(chunk)
            parallel_sfa_node.join(forked_node)
        assert_array_almost_equal(sfa_node._cov_mtx._cov_mtx, 
                                  parallel_sfa_node._cov_mtx._cov_mtx, 
                                  precision)
        sfa_node.stop_training()
        y1 = sfa_node.execute(x_test)
        parallel_sfa_node.stop_training()
        y2 = parallel_sfa_node.execute(x_test)
        assert_array_almost_equal(abs(y1), abs(y2), precision)
        
    def test_SFA2Node(self):
        """Test Parallel SFA2Node"""
        precision = 6
        x = numx_rand.random([100,10])
        x_test = numx_rand.random([20,10])
        # set different variances (avoid numerical errors)
        x *= numx.arange(1,11)
        x_test *= numx.arange(1,11)
        sfa2_node = mdp.nodes.SFA2Node()
        parallel_sfa2_node = parallel.ParallelSFA2Node()
        chunksize = 25
        chunks = [x[i*chunksize : (i+1)*chunksize] 
                    for i in range(len(x)/chunksize)]
        for chunk in chunks:
            sfa2_node.train(chunk)
            forked_node = parallel_sfa2_node.fork()
            forked_node.train(chunk)
            parallel_sfa2_node.join(forked_node)
        assert_array_almost_equal(sfa2_node._cov_mtx._cov_mtx, 
                                  parallel_sfa2_node._cov_mtx._cov_mtx, 
                                  precision)
        sfa2_node.stop_training()
        y1 = sfa2_node.execute(x_test)
        parallel_sfa2_node.stop_training()
        y2 = parallel_sfa2_node.execute(x_test)
        assert_array_almost_equal(abs(y1), abs(y2), precision)
        
    def test_FDANode(self):
        """Test Parallel FDANode."""
        # this test code is an adaption of the FDANode test
        precision = 4
        mean1 = [0., 2.]
        mean2 = [0., -2.]
        std_ = numx.array([1., 0.2])
        npoints = 50000
        rot = 45
        # input data: two distinct gaussians rotated by 45 deg
        def distr(size): 
            return numx_rand.normal(0, 1., size=(size)) * std_
        x1 = distr((npoints,2)) + mean1
        utils.rotate(x1, rot, units='degrees')
        x2 = distr((npoints,2)) + mean2
        utils.rotate(x2, rot, units='degrees')
        # labels
        cl1 = numx.ones((x1.shape[0],), dtype='d')
        cl2 = 2.*numx.ones((x2.shape[0],), dtype='d')
        flow = parallel.ParallelFlow([parallel.ParallelFDANode()])
        flow.train([[(x1, cl1), (x2, cl2)]], scheduler=parallel.Scheduler())
        fda_node = flow[0]
        assert fda_node.tlens[1] == npoints
        assert fda_node.tlens[2] == npoints
        m1 = numx.array([mean1])
        m2 = numx.array([mean2])
        utils.rotate(m1, rot, units='degrees')
        utils.rotate(m2, rot, units='degrees')
        assert_array_almost_equal(fda_node.means[1], m1, 2)
        assert_array_almost_equal(fda_node.means[2], m2, 2)
        y = flow.execute([x1, x2], scheduler=parallel.Scheduler())
        assert_array_almost_equal(numx.mean(y, axis=0), [0., 0.], precision)
        assert_array_almost_equal(numx.std(y, axis=0), [1., 1.], precision)
        assert_almost_equal(utils.mult(y[:,0], y[:,1].T), 0., precision)
        v1 = fda_node.v[:,0]/fda_node.v[0,0]
        assert_array_almost_equal(v1, [1., -1.], 2)
        v1 = fda_node.v[:,1]/fda_node.v[0,1]
        assert_array_almost_equal(v1, [1., 1.], 2)
        

def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestParallelMDPNodes))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
