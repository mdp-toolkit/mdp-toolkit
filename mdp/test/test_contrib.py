"""These are test functions for MDP contributed nodes.

Run them with:
>>> import mdp
>>> mdp.test("contrib")

"""

# import ALL stuff we use for standard nodes and delete the
# stuff we don't need. I know, this is a dirty trick.
from test_nodes import *

class ContribTestSuite(NodesTestSuite):
    def __init__(self):
        NodesTestSuite.__init__(self)
        self._cleanup_tests()

    def _set_nodes(self):
        mc = mdp.contrib
        self._nodes = [mc.JADENode]

    def _fastica_test_factory(self):
        # we don't want the fastica tests here
        pass

    def _cleanup_tests(self):
        # remove all nodes test that belong to the NodesTestSuite
        # yes, I know is a dirty trick.
        test_ids = [x.id() for x in self._tests]
        i = 0
        for test in test_ids:
            if test[:4] == "test":
                try:
                    getattr(NodesTestSuite, test)
                    # if we did not get an exception
                    # the test belongs to NodesTestSuite
                    self._tests.pop(i)
                    i -= 1
                except Exception, e:
                    pass
            i += 1

    def testJADENode(self):
        trials = 3
        for i in range(trials):
            try: 
                ica = mdp.nodes.JADENode(limit = 10**(-self.decimal))
                ica2 = ica.copy()
                self._testICANode(ica, rand_func=numx_rand.exponential)
                self._testICANodeMatrices(ica2)
                return
            except Exception, exc:
                pass
        raise exc



def get_suite():
    return ContribTestSuite()

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

