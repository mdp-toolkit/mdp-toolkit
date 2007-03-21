"""This is the test module for MDP.

Run all tests with:
>>> import mdp
>>> mdp.test()

"""

import unittest
import sys
from mdp import numx, numx_rand
import test_nodes, test_flows, test_utils, test_graph

_err_str = """\nIMPORTANT: some tests use random numbers. This could
occasionally lead to failures due to numerical degeneracies.
To rule this out, please run the tests more than once.
If you get reproducible failures please report a bug!
"""

test_suites = {'nodes': test_nodes.get_suite(),
               'flows': test_flows.get_suite(),
               'utils': test_utils.get_suite(),
               'graph': test_graph.get_suite()}

def test(suitename = 'all', verbosity = 2, seed = None):
    if seed is None:
        seed = int(numx_rand.randint(2**31-1))

    numx_rand.seed(seed)
        
    sys.stderr.write("Random Seed: " + str(seed)+'\n')
    if suitename == 'all':
        suite = unittest.TestSuite(test_suites.values())
    else:
        suite = test_suites[suitename]
    res = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    if len(res.errors+res.failures) > 0:
        sys.stderr.write(_err_str)
    sys.stderr.write("\nRandom Seed was: " + str(seed)+'\n')
    
