"""This is the test module for MDP.

Run all tests with:
>>> import mdp
>>> mdp.test.test()

Run all benchmarks with:
>>> import mdp
>>> mdp.test.benchmark()

"""

import unittest
import sys
from mdp import numx, numx_rand
import test_nodes, test_flows, test_utils, test_graph
import benchmark_mdp
from testing_tools import run_benchmarks

test_suites = {'nodes': test_nodes.get_suite(),
               'flows': test_flows.get_suite(),
               'utils': test_utils.get_suite(),
               'graph': test_graph.get_suite()}

def test(suitename = 'all', verbosity = 2, seed = None):
    if seed is None:
        #numx_rand.seed(1268049219)
        pass
    else:
        numx_rand.seed(seed)
    sys.stderr.write("Random Seed: " + str(numx_rand.get_state()[1][0])+'\n')
    if suitename == 'all':
        suite = unittest.TestSuite(test_suites.values())
    else:
        suite = test_suites[suitename]
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

benchmark_suites = {'mdp': benchmark_mdp.get_benchmarks()}
    
def benchmark(suitename = 'all'):
    numx_rand.seed(1268049219)
    if suitename == 'all':
        suite = []
        for s in benchmark_suites.itervalues(): suite.extend(s)
    else:
        suite = benchmark_suites[suitename]
    print "Random Seed: ", numx_rand.get_state()[1][0]
    run_benchmarks(suite)
