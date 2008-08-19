
import unittest

import mdp
import mdp.parallel as parallel
n = mdp.numx
import numpy.testing as testing_tools


class TestOrderedExcecution(unittest.TestCase):

    def test_ordered(self):
        """Test ordered parallel execution by running some jobs."""
        ## train flow to prepare for execution ##
        shape = (20,10)
        output_dim = 7
        flow = parallel.ParallelFlow([
                            parallel.ParallelSFANode(output_dim=5),
                            mdp.nodes.PolynomialExpansionNode(degree=3),
                            parallel.ParallelSFANode(output_dim=output_dim)])
        data_generators = [n.random.random((6,)+shape), 
                           None, 
                           n.random.random((6,)+shape)]
        flow.parallel_train(data_generators)
        while flow.is_parallel_training():
            results = []
            while flow.job_available():
                job = flow.get_job()
                results.append(job())
            flow.use_results(results)
        ## test ordered parallel execution ###
        scheduler = parallel.SimpleScheduler(result_container=
                                        parallel.OrderedListResultContainer())
        shape = [20,10]  # mark first job by consisting of zeros
        data_iter = [n.zeros(shape)]
        for _ in range(6):
            data_iter.append(n.random.random(shape))
        data_iter = parallel.OrderedIterable(data_iter)
        flow.parallel_execute(data_iter, 
                              execute_job_class=parallel.OrderedFlowExecuteJob)
        first_job = flow.get_job()
        while flow.job_available():
            job = flow.get_job()
            scheduler.add_job(job)
        # process the first job last to mess up the order
        scheduler.add_job(first_job)
        results = scheduler.get_results()
        result_array = flow.use_results(results)
        # test if the first entries are really zeros
        precision = 10
        parallel_res = result_array[:shape[0]]
        direct_res = flow.execute(n.zeros(shape))
        testing_tools.assert_array_almost_equal(parallel_res, direct_res,
                                                precision)

    
def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestOrderedExcecution))
    return suite
            
if __name__ == '__main__':
    unittest.main() 