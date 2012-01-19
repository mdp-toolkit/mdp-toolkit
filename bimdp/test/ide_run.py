"""
Helper script to run or debug the tests in an IDE as a simple .py file.
"""

import py

#args_str = ""
args_str = "-k parallel --maxfail 1 --tb native"
#args_str = "--maxfail 1 --tb native"

py.test.cmdline.main(args_str.split(" "))
