"""
Helper script to run or debug the tests in an IDE as a simple .py file.
"""

import py

#args_str = ""
args_str = "-k parallel --maxfail 1 --tb long -v"

py.test.cmdline.main(args_str.split(" "))