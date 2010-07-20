"""This is the test module for MDP.

Run all tests with:
>>> import mdp
>>> mdp.test()

"""

import unittest
import sys
import mdp

import test_nodes
import test_flows
import test_utils
import test_graph
import test_contrib
import test_extension
import test_hinet
import test_schedule
import test_parallelnodes
import test_parallelflows
import test_parallelhinet
import test_process_schedule
import test_classifier

import test_shutdown

numx = mdp.numx
numx_rand = mdp.numx_rand

_err_str = """\nIMPORTANT: some tests use random numbers. This could
occasionally lead to failures due to numerical degeneracies.
To rule this out, please run the tests more than once.
If you get reproducible failures please report a bug!
"""

test_suites = {'flows': (test_flows.get_suite, 0),
               'utils': (test_utils.get_suite, 1),
               'graph': (test_graph.get_suite, 2),
               'nodes': (test_nodes.get_suite, 3),
               'extension': (test_extension.get_suite, 4),
               'hinet':   (test_hinet.get_suite, 5),
               'schedule': (test_schedule.get_suite, 6),
               'parallelnodes': (test_parallelnodes.get_suite, 7),
               'parallelflows': (test_parallelflows.get_suite, 8),
               'parallelhinet': (test_parallelhinet.get_suite, 9),
               'process_schedule': (test_process_schedule.get_suite, 10),
               'contrib': (test_contrib.get_suite, 11),
               'classifier': (test_classifier.get_suite, 12)}
                           

def test(suitename = 'all', verbosity = 2, seed = None, testname = None):
    if seed is None:
        seed = int(numx_rand.randint(2**31-1))

    numx_rand.seed(seed)
    mdp.info()
    sys.stderr.write("\nRandom Seed: " + str(seed)+'\n')
    if suitename == 'all':
        sorted_suites = [x[0](testname=testname)
                         for x in sorted(test_suites.values(),
                                         key=lambda y: y[1])]
        suite = unittest.TestSuite(sorted_suites)
    else:
        suite = test_suites[suitename][0](testname=testname)
    res = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    if len(res.errors+res.failures) > 0:
        sys.stderr.write(_err_str)
    mdp.info()
    sys.stderr.write("\nRandom Seed was: " + str(seed)+'\n')
    
