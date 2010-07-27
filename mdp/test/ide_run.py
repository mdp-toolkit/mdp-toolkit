"""
Helper script to run or debug the tests in an IDE as a simple .py file.
"""

import py

args_str = "-k test_hinet"

py.test.cmdline.main(args_str.split(" "))