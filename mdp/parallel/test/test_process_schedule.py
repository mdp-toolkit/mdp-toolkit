
import unittest

import mdp
import mdp.parallel as parallel
n = mdp.numx

class TestScheduler(unittest.TestCase):

    def test_scheduler(self):
        """Test process scheduler with 6 jobs and 2 processes."""
        scheduler = parallel.ProcessScheduler(verbose=False, n_processes=2)
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