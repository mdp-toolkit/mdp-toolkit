"""
Helper script to run or debug the tests in an IDE as a simple .py file.
"""

import py

#args_str = ""
args_str = "-k test_nodes_generic"

py.test.cmdline.main(args_str.split(" "))