import unittest

import mdp.parallel as parallel
from mdp import numx as n


class TestProcessScheduler(unittest.TestCase):

    def test_scheduler(self):
        """Test process scheduler with 6 jobs and 3 processes."""
        ## get MDP source path (if not in the pythonpath) 
        # module_file = os.path.abspath(inspect.getfile(sys._getframe(0)))
        # module_path = os.path.dirname(module_file)
        # source_paths = [os.path.join(module_path.split("mdp")[0])[0:-1],]
        scheduler = parallel.ProcessScheduler(verbose=False, 
                                              n_processes=3,
                                              source_paths=None)
        for i in range(6):
            scheduler.add_task(i, parallel.SqrTestCallable())
        results = scheduler.get_results()
        scheduler.shutdown()
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
