
import unittest

import numpy as n

import mdp.parallel as parallel
import mdp.parallel.pp_support as pp_support


class TestLocalPPScheduler(unittest.TestCase):

    def test_simple(self):
        """Test local scheduling."""
        scheduler = pp_support.LocalPPScheduler(ncpus=2,
                                                max_queue_length=0, 
                                                verbose=False)
        # process jobs
        for i in range(50):
            scheduler.add_task(i, parallel.SqrTestCallable())
        results = scheduler.get_results()
        scheduler.cleanup()
        # check result
        results.sort()
        results = n.array(results[:6])
        self.assertTrue(n.all(results == n.array([0,1,4,9,16,25])))
        

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLocalPPScheduler))
    return suite
            
if __name__ == '__main__':
    unittest.main() 

