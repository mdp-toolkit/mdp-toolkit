
import unittest

import numpy as n

import mdp.parallel as parallel
import mdp.parallel.pp_support as pp_support


remote_slaves = [("sherrington", 1),
                 ("weismann", 2)]
                  
python_executable = "/home/wilbert/bin/python"
sys_paths = ["/home/wilbert/develop/workspace/MDP"]



class TestRemotePPScheduler(unittest.TestCase):

    def test_simple(self):
        """Test."""
        scheduler = pp_support.NetworkPPScheduler(
                                    remote_slaves=remote_slaves, 
                                    timeout=60, 
                                    source_paths=sys_paths, 
                                    remote_python_executable=python_executable,
                                    verbose=True)
        # process jobs
        for i in range(30):
            scheduler.add_task(i, parallel.SqrTestCallable())
        results = scheduler.get_results()
        scheduler.cleanup()
        # check result
        results.sort()
        results = n.array(results)
        self.assertTrue(n.all(results[:6] == n.array([0,1,4,9,16,25])))
        
       
def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestRemotePPScheduler))
    return suite
            
if __name__ == '__main__':
    unittest.main() 

