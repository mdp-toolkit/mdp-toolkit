
import unittest
import sys

import test_parallelnodes
import test_parallelflows
import test_parallelhinet
import test_resultorder
import test_process_schedule

test_suites = {"parallelnodes": test_parallelnodes.get_suite(),
               "parallelflows": test_parallelflows.get_suite(),
               "parallelhinet": test_parallelhinet.get_suite(),
               "resultorder": test_resultorder.get_suite(),
               "process_schedule": test_process_schedule.get_suite()}

def get_suite():
    return unittest.TestSuite(test_suites.values())

def test(suitename = 'all'):
    if suitename == 'all':
        suite = unittest.TestSuite(test_suites.values())
    else:
        suite = test_suites[suitename]
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if len(res.errors+res.failures) > 0:
        sys.stderr.write(res._err_str)

if __name__ == '__main__':
    test()