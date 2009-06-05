import unittest

import mdp
import mdp.parallel as parallel
from mdp import numx as n

import testing_tools


class TestProcessScheduler(unittest.TestCase):

# this test has been superseded by test_scheduler_order, skip this for speed
#    def test_scheduler(self):
#        """Test process scheduler with 8 tasks and 3 processes."""
#        scheduler = parallel.ProcessScheduler(verbose=False, 
#                                              n_processes=3,
#                                              source_paths=None)
#        for i in range(8):
#            scheduler.add_task(i, parallel._SqrTestCallable())
#        results = scheduler.get_results()
#        scheduler.shutdown()
#        # check result
#        results = n.array(results)
#        self.assertTrue(n.all(results == n.array([0,1,4,9,16,25,36,49])))
        
    def test_scheduler_order(self):
        """Test the correct result order in process scheduler."""
        scheduler = parallel.ProcessScheduler(verbose=False, 
                                              n_processes=3,
                                              source_paths=None)
        max_i = 8
        for i in range(max_i):
            scheduler.add_task((n.arange(0,i+1), (max_i-1-i)*1.0/4),
                               parallel.SleepSqrTestCallable())
        results = scheduler.get_results()
        scheduler.shutdown()
        # check result
        results = n.concatenate(results)
        self.assertTrue(n.all(results ==
                              n.concatenate([n.arange(0,i+1)**2
                                             for i in range(max_i)])))
        
    def test_scheduler_no_cache(self):
        """Test process scheduler with caching turned off."""
        scheduler = parallel.ProcessScheduler(verbose=False, 
                                              n_processes=2,
                                              source_paths=None,
                                              cache_callable=False)
        for i in range(8):
            scheduler.add_task(i, parallel.SqrTestCallable())
        results = scheduler.get_results()
        scheduler.shutdown()
        # check result
        results = n.array(results)
        self.assertTrue(n.all(results == n.array([0,1,4,9,16,25,36,49])))
        
    def test_scheduler_makeparallel1(self):
        """Test process scheduler with real Nodes."""
        precision = 6
        node1 = mdp.nodes.PCANode(output_dim=20)
        node2 = mdp.nodes.PolynomialExpansionNode(degree=1)
        node3 = mdp.nodes.SFANode(output_dim=10)
        flow = mdp.Flow([node1, node2, node3])
        parallel_flow = parallel.make_flow_parallel(flow.copy())
        scheduler = parallel.ProcessScheduler(verbose=False, 
                                              n_processes=3,
                                              source_paths=None)
        input_dim = 30
        scales = n.linspace(1, 100, num=input_dim)
        scale_matrix = mdp.numx.diag(scales)
        train_iterables = [n.dot(mdp.numx_rand.random((5, 100, input_dim)),
                                 scale_matrix) 
                           for _ in range(3)]
        parallel_flow.train(train_iterables, scheduler=scheduler)
        scheduler.shutdown()
        flow.train(train_iterables)
        self.assertTrue(parallel_flow[0].tlen == flow[0].tlen)
        reconstructed_flow = parallel.unmake_flow_parallel(parallel_flow)
        x = mdp.numx.random.random((10, input_dim))
        y1 = flow.execute(x)
        y2 = reconstructed_flow.execute(x)
        testing_tools.assert_array_almost_equal(abs(y1), abs(y2), precision)
        

def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestProcessScheduler))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
