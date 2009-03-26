import unittest

import mdp.parallel as parallel
from mdp import numx as n


class TestProcessScheduler(unittest.TestCase):

    def test_scheduler(self):
        """Test process scheduler with 8 tasks and 3 processes."""
        scheduler = parallel.ProcessScheduler(verbose=False, 
                                              n_processes=3,
                                              source_paths=None)
        for i in range(8):
            scheduler.add_task(i, parallel.SqrTestCallable())
        results = scheduler.get_results()
        scheduler.shutdown()
        # check result
        results = n.array(results)
        self.assertTrue(n.all(results == n.array([0,1,4,9,16,25,36,49])))
        
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
        

def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestProcessScheduler))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
