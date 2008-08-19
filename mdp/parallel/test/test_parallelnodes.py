
import unittest

import mdp
import mdp.parallel as parallel
n = mdp.numx
import numpy.testing as testing_tools


class TestParallelMDPNodes(unittest.TestCase):
    
    def test_PCANode(self):
        precision = 10
        x = n.random.random([100,10])
        x_test = n.random.random([20,10])
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
        testing_tools.assert_array_almost_equal(
                                        pca_node._cov_mtx._cov_mtx, 
                                        parallel_pca_node._cov_mtx._cov_mtx, 
                                        precision)
        pca_node.stop_training()
        y1 = pca_node.execute(x_test)
        parallel_pca_node.stop_training()
        y2 = parallel_pca_node.execute(x_test)
        testing_tools.assert_array_almost_equal(y1, y2, precision)
        
    def test_WhiteningNode(self):
        precision = 10
        x = n.random.random([100,10])
        x_test = n.random.random([20,10])
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
        testing_tools.assert_array_almost_equal(
                                        pca_node._cov_mtx._cov_mtx, 
                                        parallel_pca_node._cov_mtx._cov_mtx, 
                                        precision)
        pca_node.stop_training()
        y1 = pca_node.execute(x_test)
        parallel_pca_node.stop_training()
        y2 = parallel_pca_node.execute(x_test)
        testing_tools.assert_array_almost_equal(y1, y2, precision)

    def test_SFANode(self):
        precision = 10
        x = n.random.random([100,10])
        x_test = n.random.random([20,10])
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
        testing_tools.assert_array_almost_equal(
                                        sfa_node._cov_mtx._cov_mtx, 
                                        parallel_sfa_node._cov_mtx._cov_mtx, 
                                        precision)
        sfa_node.stop_training()
        y1 = sfa_node.execute(x_test)
        parallel_sfa_node.stop_training()
        y2 = parallel_sfa_node.execute(x_test)
        testing_tools.assert_array_almost_equal(y1, y2, precision)
        
    def test_SFA2Node(self):
        precision = 10
        x = n.random.random([100,10])
        x_test = n.random.random([20,10])
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
        testing_tools.assert_array_almost_equal(
                                        sfa2_node._cov_mtx._cov_mtx, 
                                        parallel_sfa2_node._cov_mtx._cov_mtx, 
                                        precision)
        sfa2_node.stop_training()
        y1 = sfa2_node.execute(x_test)
        parallel_sfa2_node.stop_training()
        y2 = parallel_sfa2_node.execute(x_test)
        testing_tools.assert_array_almost_equal(y1, y2, precision)
        

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestParallelMDPNodes))
    return suite
            
if __name__ == '__main__':
    unittest.main() 