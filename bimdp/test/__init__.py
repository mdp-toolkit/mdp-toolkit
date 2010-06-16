
import unittest
import sys, os

import test_binode
import test_biflow
import test_bihinet
import test_parallelbiflow
import test_parallelbihinet

test_suites = {
    "binode": test_binode.get_suite(),
    "biflow": test_biflow.get_suite(),
    "bihinet": test_bihinet.get_suite(),
    "parallelbiflow": test_parallelbiflow.get_suite(),
    "parallelbihinet": test_parallelbihinet.get_suite(),
}

def get_suite():
    return unittest.TestSuite(test_suites.values())

def test(suitename = 'all'):
    if suitename == 'all':
        suite = unittest.TestSuite(test_suites.values())
    else:
        suite = test_suites[suitename]
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    test()