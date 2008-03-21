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
        self.mat_dim = (500,4)
        self._cleanup_tests()

    def _set_nodes(self):
        self._nodes = [mc.JADENode,
                       mc.NIPALSNode,
                       (mc.FlowNode, [self._get_new_flow], None),
                       (mc.Layer, [self._get_new_nodes], None),
                       (mc.CloneLayer, [self._get_sigle_node, 2], None),]

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

    def _get_new_flow(self):
        flow = mdp.Flow([mdp.nodes.NoiseNode(), 
                         mdp.nodes.SFANode()])
        return flow

    def _get_new_nodes(self):
        node1 = mdp.nodes.CuBICANode(input_dim=1, whitened=True)
        node2 = mdp.nodes.CuBICANode(input_dim=2, whitened=True)
        node3 = mdp.nodes.CuBICANode(input_dim=1, whitened=True)
        return [node1, node2, node3]

    def _get_sigle_node(self): 
        node1 = mdp.nodes.CuBICANode(input_dim=2, whitened=True)
        return node1

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

    def testFlowNode_training(self):
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2), 
                         mdp.nodes.PCANode(output_dim=15, reduce=True),
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.PCANode(output_dim=3, reduce=True)])
        flownode = mc.FlowNode(flow)
        x = numx_rand.random([300,20])
        while flownode.get_remaining_train_phase() > 0:
            flownode.train(x)
            flownode.stop_training()
        flownode.execute(x)

    def testFlowNode_trainability(self):
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2)])
        flownode = mc.FlowNode(flow)
        assert flownode.is_trainable() is False
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2), 
                         mdp.nodes.PCANode(output_dim=15),
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.PCANode(output_dim=3)])
        flownode = mc.FlowNode(flow)
        assert flownode.is_trainable() is True
        
    def testFlowNode_invertibility(self):
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2)])
        flownode = mc.FlowNode(flow)
        assert flownode.is_invertible() is False
        flow = mdp.Flow([mdp.nodes.PCANode(output_dim=15),
                         mdp.nodes.SFANode(),
                         mdp.nodes.PCANode(output_dim=3)])
        flownode = mc.FlowNode(flow)
        assert flownode.is_invertible() is True
    
    def testFlowNode_pretrained_node(self):
        x = numx_rand.random([100,10])
        pretrained_node = mdp.nodes.PCANode(output_dim=6)
        pretrained_node.train(x)
        pretrained_node.stop_training()
        flow = mdp.Flow([pretrained_node,
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.PCANode(output_dim=3)])
        flownode = mc.FlowNode(flow)
        while flownode.get_remaining_train_phase() > 0:
            flownode.train(x)
            flownode.stop_training()
        flownode.execute(x)

    def testLayer(self):
        node1 = mdp.nodes.PCANode(input_dim=10, output_dim=5)
        node2 = mdp.nodes.PCANode(input_dim=17, output_dim=3)
        node3 = mdp.nodes.PCANode(input_dim=3, output_dim=1)
        x = numx_rand.random([100,30]).astype('f')
        layer = mc.Layer([node1, node2, node3])
        layer.train(x)
        y = layer.execute(x)
        assert layer.dtype == numx.dtype('f')
        assert y.dtype == layer.dtype

    def testCloneLayer(self):
        node = mdp.nodes.PCANode(input_dim=10, output_dim=5)
        x = numx_rand.random([10,70]).astype('f')
        layer = mc.CloneLayer(node, 7)
        layer.train(x)
        y = layer.execute(x)
        assert layer.dtype == numx.dtype('f')
        assert y.dtype == layer.dtype

    def testSwitchboardRouting1(self):
        sboard = mc.Rectangular2dSwitchboard(x_in_channels=3, 
                                             y_in_channels=2,
                                             in_channel_dim=2,
                                             x_field_channels=2, 
                                             y_field_channels=1,
                                             x_field_spacing=1, 
                                             y_field_spacing=1)
        assert numx.all(sboard.connections == 
                               numx.array([0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 
                                           8, 9, 8, 9, 10, 11])) == True
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        # test generated switchboard
        channel_sboard = sboard.get_out_channel_node(0)
        channel_sboard.execute(x)

    def testSwitchboardRouting2(self):
        sboard = mc.Rectangular2dSwitchboard(x_in_channels=2, 
                                             y_in_channels=4, 
                                             in_channel_dim=1,
                                             x_field_channels=1, 
                                             y_field_channels=2,
                                             x_field_spacing=1, 
                                             y_field_spacing=2)
        assert numx.all(sboard.connections == 
                        numx.array([0, 2, 1, 3, 4, 6, 5, 7])) == True
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        # test generated switchboard
        channel_sboard = sboard.get_out_channel_node(0)
        channel_sboard.execute(x)
        
    def testSwitchboard_get_out_channel_node(self):
        sboard = mc.Rectangular2dSwitchboard(x_in_channels=5, 
                                             y_in_channels=4,
                                             in_channel_dim=2,
                                             x_field_channels=3, 
                                             y_field_channels=2,
                                             x_field_spacing=1, 
                                             y_field_spacing=2)
        x = numx.array([range(0, sboard.input_dim), 
                     range(101, 101+sboard.input_dim)])
        y = sboard.execute(x)
        # routing layer
        nodes = [sboard.get_out_channel_node(index) 
                 for index in range(sboard.output_channels)]
        layer = mc.Layer(nodes, same_input=True)
        layer_y = layer.execute(x)
        assert numx.all(y==layer_y) == True

    def testHinetSimpleNet(self):
        switchboard = mc.Rectangular2dSwitchboard(x_in_channels=12, 
                                                  y_in_channels=8,
                                                  x_field_channels=4, 
                                                  y_field_channels=4,
                                                  x_field_spacing=2, 
                                                  y_field_spacing=2,
                                                  in_channel_dim=3)
        
        node = mdp.nodes.PCANode(input_dim=4*4*3, output_dim=5)
        flownode = mc.FlowNode(mdp.Flow([node,]))
        layer = mc.CloneLayer(flownode, switchboard.output_channels)
        flow = mdp.Flow([switchboard, layer])
        x = numx_rand.random([5, switchboard.input_dim])
        flow.train(x)

    def testSFANet(self):
        noisenode = mdp.nodes.NoiseNode(input_dim=20*20, 
                                        noise_args=(0, 0.0001))
        sfa_node = mdp.nodes.SFANode(input_dim=20*20, output_dim=10,dtype='f')
        switchboard = mc.Rectangular2dSwitchboard(x_in_channels=100, 
                                                  y_in_channels=100,
                                                  x_field_channels=20, 
                                                  y_field_channels=20,
                                                  x_field_spacing=10, 
                                                  y_field_spacing=10)
        flownode = mc.FlowNode(mdp.Flow([noisenode, sfa_node]))
        sfa_layer = mc.CloneLayer(flownode, switchboard.output_channels)
        flow = mdp.Flow([switchboard, sfa_layer])
        train_gen = numx_rand.random((3, 10, 100*100))
        flow.train([None, train_gen])
            
def get_suite(testname=None):
    return ContribTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())

