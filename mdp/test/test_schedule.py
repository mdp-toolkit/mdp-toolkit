import unittest

import mdp
import mdp.parallel as parallel
from mdp import numx as n

import testing_tools


class TestScheduler(unittest.TestCase):

    def test_scheduler(self):
        """Test scheduler with 6 tasks."""
        scheduler = parallel.Scheduler()
        for i in range(6):
            scheduler.add_task(i, lambda x: x**2)
        results = scheduler.get_results()
        scheduler.shutdown()
        # check result
        results = n.array(results)
        self.assertTrue(n.all(results == n.array([0,1,4,9,16,25])))
    
        
class TestThreadScheduler(unittest.TestCase):

    def test_scheduler_flow(self):
        """Test thread scheduler with real Nodes."""
        precision = 6
        node1 = mdp.nodes.PCANode(output_dim=20)
        node2 = mdp.nodes.PolynomialExpansionNode(degree=1)
        node3 = mdp.nodes.SFANode(output_dim=10)
        flow = mdp.parallel.ParallelFlow([node1, node2, node3])
        parallel_flow = mdp.parallel.ParallelFlow(flow.copy()[:])
        scheduler = parallel.ThreadScheduler(verbose=False, 
                                             n_threads=3)
        input_dim = 30
        scales = n.linspace(1, 100, num=input_dim)
        scale_matrix = mdp.numx.diag(scales)
        train_iterables = [n.dot(mdp.numx_rand.random((5, 100, input_dim)),
                                 scale_matrix) 
                           for _ in range(3)]
        parallel_flow.train(train_iterables, scheduler=scheduler)
        x = mdp.numx.random.random((10, input_dim))
        # test that parallel execution works as well
        # note that we need more chungs then processes to test caching
        parallel_flow.execute([x for _ in range(8)], scheduler=scheduler)
        scheduler.shutdown()
        # compare to normal flow
        flow.train(train_iterables)
        self.assertTrue(parallel_flow[0].tlen == flow[0].tlen)
        y1 = flow.execute(x)
        y2 = parallel_flow.execute(x)
        testing_tools.assert_array_almost_equal(abs(y1), abs(y2), precision)
        

def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestScheduler))
    suite.addTest(unittest.makeSuite(TestThreadScheduler))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
