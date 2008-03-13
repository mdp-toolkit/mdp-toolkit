"""This is the test module for MDP.

Run all tests with:
>>> import mdp
>>> mdp.test()

"""

import unittest
import sys
from mdp import numx, numx_rand
import test_nodes, test_flows, test_utils, test_graph, test_contrib

_err_str = """\nIMPORTANT: some tests use random numbers. This could
occasionally lead to failures due to numerical degeneracies.
To rule this out, please run the tests more than once.
If you get reproducible failures please report a bug!
"""

test_suites = {'nodes':   (test_nodes.get_suite(),   3),
               'contrib': (test_contrib.get_suite(), 4),
               'flows':   (test_flows.get_suite(),   0),
               'utils':   (test_utils.get_suite(),   1),
               'graph':   (test_graph.get_suite(),   2)}

def test(suitename = 'all', verbosity = 2, seed = None):
    if seed is None:
        seed = int(numx_rand.randint(2**31-1))

    numx_rand.seed(seed)
        
    sys.stderr.write("Random Seed: " + str(seed)+'\n')
    if suitename == 'all':
        sorted_suites = [x[0] for x in sorted(test_suites.values(),
                                              key=lambda y: y[1])]
        suite = unittest.TestSuite(sorted_suites)
    else:
        suite = test_suites[suitename][0]
    res = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    if len(res.errors+res.failures) > 0:
        sys.stderr.write(_err_str)
    sys.stderr.write("\nRandom Seed was: " + str(seed)+'\n')
    
