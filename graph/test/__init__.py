"""This is the test module for graph.

Run all tests with:
>>> import graph
>>> graph.test.test()

"""

import unittest
import test_graph

def test(verbosity = 2):
    suite = test_graph.get_suite()
    unittest.TextTestRunner(verbosity=verbosity).run(suite)
