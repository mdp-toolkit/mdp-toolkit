"""These are test functions for MDP utilities.

Run them with:
>>> import mdp
>>> mdp.test.test("utils")

"""
import unittest
import pickle
import os
import tempfile
from mdp import utils, numx_rand

class UtilsTestCase(unittest.TestCase):
##     def testProgressBar(self):
##         print
##         p = utils.ProgressBar(minimum=0,maximum=1000)
##         for i in range(1000):
##             p.update(i+1)
##             for j in xrange(10000): pass
##         print

    def testCrashRecoveryException(self):
        a = 3
        try:
            raise utils.CrashRecoveryException, \
                  ('bogus errstr',a,StandardError())
        except utils.CrashRecoveryException, e:
            filename1 = e.dump()
            filename2 = e.dump(os.path.join(tempfile.gettempdir(),'removeme'))
            assert isinstance(e.parent_exception, StandardError)

        for fname in [filename1,filename2]:
            fl = file(fname)
            obj = pickle.load(fl)
            fl.close()
            os.remove(fname)
            assert obj == a
            
        
def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(UtilsTestCase))
    return suite


if __name__ == '__main__':
    numx_rand.seed(1268049219, 2102953867)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
