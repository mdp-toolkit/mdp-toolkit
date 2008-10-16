
import unittest

import mdp.parallel as parallel
from mdp import numx as n


class TestProcessScheduler(unittest.TestCase):

    def test_scheduler(self):
        """Test process scheduler with 6 jobs and 3 processes."""
        scheduler = parallel.Scheduler()
        for i in range(6):
            scheduler.add_task(i, lambda x: x**2)
        results = scheduler.get_results()
        scheduler.cleanup()
        # check result
        results = n.array(results)
        self.assertTrue(n.all(results == n.array([0,1,4,9,16,25])))
        

def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestProcessScheduler))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
