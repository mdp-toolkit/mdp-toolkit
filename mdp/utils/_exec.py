"""Minimalistic helper to overcome problems with exec on elderly Python 2.7"""

# There is an issue with exec(st, g, l) not being treated correctly with python
# 2.7.3 on precise/travis.  Since it spits SyntaxError, decision between the two
# needs to be done at code parsing stage thus here we define the adapter function

def _exec(st, g, l):
    exec st in g, l
