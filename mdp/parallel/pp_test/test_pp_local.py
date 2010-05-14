
import unittest

import numpy as n

import mdp
import mdp.parallel as parallel
from mdp.parallel import pp_support


class TestLocalPPScheduler(unittest.TestCase):

    def test_simple(self):
        """Test local pp scheduling."""
        scheduler = pp_support.LocalPPScheduler(ncpus=2,
                                                max_queue_length=0, 
                                                verbose=False)
        # process jobs
        for i in range(50):
            scheduler.add_task(i, parallel.SqrTestCallable())
        results = scheduler.get_results()
        scheduler.shutdown()
        # check result
        results.sort()
        results = n.array(results[:6])
        self.assertTrue(n.all(results == n.array([0,1,4,9,16,25])))
        
    def test_scheduler_flow(self):
        """Test local pp scheduler with real Nodes."""
        precision = 6
        node1 = mdp.nodes.PCANode(output_dim=20)
        node2 = mdp.nodes.PolynomialExpansionNode(degree=1)
        node3 = mdp.nodes.SFANode(output_dim=10)
        flow = mdp.parallel.ParallelFlow([node1, node2, node3])
        parallel_flow = mdp.parallel.ParallelFlow(flow.copy()[:])
        scheduler = pp_support.LocalPPScheduler(ncpus=3,
                                                max_queue_length=0, 
                                                verbose=False)
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
        self.assert_(n.max(n.abs(y1 - y2)) < 10**(-precision))
        

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLocalPPScheduler))
    return suite
            
if __name__ == '__main__':
    unittest.main() 

