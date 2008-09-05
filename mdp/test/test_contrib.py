"""These are test functions for MDP contributed nodes.

Run them with:
>>> import mdp
>>> mdp.test("contrib")

"""

# import ALL stuff we use for standard nodes and delete the
# stuff we don't need. I know, this is a dirty trick.
from test_nodes import *

mc = mdp.contrib

import itertools    


class ContribTestSuite(NodesTestSuite):
    def __init__(self, testname=None):
        NodesTestSuite.__init__(self, testname=testname)
        self.mat_dim = (500, 5)
        self._cleanup_tests()

    def _set_nodes(self):
        self._nodes = [mc.JADENode,
                       mc.NIPALSNode,
                       (mc.LLENode, [3, 0.001, False], None),
                       (mc.LLENode, [3, 0.001, True], None)]

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
    
    def testNIPALSNode(self):
        line_x = numx.zeros((1000,2),"d")
        line_y = numx.zeros((1000,2),"d")
        line_x[:,0] = numx.linspace(-1,1,num=1000,endpoint=1)
        line_y[:,1] = numx.linspace(-0.2,0.2,num=1000,endpoint=1)
        mat = numx.concatenate((line_x,line_y))
        des_var = std(mat,axis=0)
        utils.rotate(mat,uniform()*2*numx.pi)
        mat += uniform(2)
        pca = mdp.nodes.NIPALSNode(conv=1E-15, max_it=1000)
        pca.train(mat)
        act_mat = pca.execute(mat)
        assert_array_almost_equal(mean(act_mat,axis=0),\
                                  [0,0],self.decimal)
        assert_array_almost_equal(std(act_mat,axis=0),\
                                  des_var,self.decimal)
        # test a bug in v.1.1.1, should not crash
        pca.inverse(act_mat[:,:1])
        # try standard PCA on the same data and compare the eigenvalues
        pca2 = mdp.nodes.PCANode()
        pca2.train(mat)
        pca2.stop_training()
        assert_array_almost_equal(pca2.d, pca.d, self.decimal)
        
    def testNIPALSNode_desired_variance(self):
        mat, mix, inp = self._get_random_mix(mat_dim=(1000, 3))
        # first make them white
        pca = mdp.nodes.WhiteningNode()
        pca.train(mat)
        mat = pca.execute(mat)
        # set the variances
        mat *= [0.6,0.3,0.1]
        #mat -= mat.mean(axis=0)
        pca = mdp.nodes.NIPALSNode(output_dim=0.8)
        pca.train(mat)
        out = pca.execute(mat)
        # check that we got exactly two output_dim:
        assert pca.output_dim == 2
        assert out.shape[1] == 2
        # check that explained variance is > 0.8 and < 1
        assert (pca.explained_variance > 0.8 and pca.explained_variance < 1)

        
def get_suite(testname=None):
    return ContribTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

