"""
Helper script to run or debug the tests in an IDE as a simple .py file.
"""

import py

#args_str = ""
#args_str = "-k hinet --maxfail 1 -s"
args_str = "--maxfail 1"

py.test.cmdline.main(args_str.split(" "))