"""
Helper script to run or debug the tests in an IDE as a simple .py file.
"""

import pytest

#args_str = ""
args_str = "-k hinet --maxfail 1 -s --tb native"
#args_str = "--maxfail 1 --tb native"

pytest.cmdline.main(args_str.split(" "))
