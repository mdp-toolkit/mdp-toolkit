
import unittest

import os
import sys
import inspect

import mdp
import mdp.parallel as parallel
from mdp import numx as n


class TestScheduler(unittest.TestCase):

    def test_scheduler(self):
        """Test process scheduler with 6 jobs and 3 processes."""
        ## get MDP source path (it might not be installed in the pythonpath) 
        # module_file = os.path.abspath(inspect.getfile(sys._getframe(0)))
        # module_path = os.path.dirname(module_file)
        # mdp_path = os.path.join(module_path.split("mdp")[0])[0:-1]
        scheduler = parallel.ProcessScheduler(verbose=False, n_processes=3,
                                              source_paths=None)
        for i in range(6):
            job = parallel.TestJob(x=i)
            scheduler.add_job(job)
        results = scheduler.get_results()
        scheduler.cleanup()
        # check result
        results.sort()
        results = n.array(results)
        self.assertTrue(n.all(results == n.array([0,1,4,9,16,25])))
        

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestScheduler))
    return suite
            
if __name__ == '__main__':
    unittest.main() 