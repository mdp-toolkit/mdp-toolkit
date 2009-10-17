"""These are test functions for hinet.

Run them with:
>>> import mdp
>>> mdp.test("hinet")

"""

# import ALL stuff we use for standard nodes and delete the
# stuff we don't need. I know, this is a dirty trick.
from test_nodes import *

import StringIO

mh = mdp.hinet

class HinetTestSuite(NodesTestSuite):
    
    def __init__(self, testname=None):
        NodesTestSuite.__init__(self, testname=testname)
        self.mat_dim = (500,4)
        self._cleanup_tests()

    def _set_nodes(self):
        self._nodes = [(mh.FlowNode, [self._get_new_flow], None),
                       (mh.Layer, [self._get_new_nodes], None),
                       (mh.CloneLayer, [self._get_sigle_node, 2], None),]

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

    def testFlowNode_training(self):
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2), 
                         mdp.nodes.PCANode(output_dim=15, reduce=True),
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.PCANode(output_dim=3, reduce=True)])
        flownode = mh.FlowNode(flow)
        x = numx_rand.random([300,20])
        while flownode.get_remaining_train_phase() > 0:
            flownode.train(x)
            flownode.stop_training()
        flownode.execute(x)

    def testFlowNode_trainability(self):
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2)])
        flownode = mh.FlowNode(flow)
        assert flownode.is_trainable() is False
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2), 
                         mdp.nodes.PCANode(output_dim=15),
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.PCANode(output_dim=3)])
        flownode = mh.FlowNode(flow)
        assert flownode.is_trainable() is True
        
    def testFlowNode_invertibility(self):
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2)])
        flownode = mh.FlowNode(flow)
        assert flownode.is_invertible() is False
        flow = mdp.Flow([mdp.nodes.PCANode(output_dim=15),
                         mdp.nodes.SFANode(),
                         mdp.nodes.PCANode(output_dim=3)])
        flownode = mh.FlowNode(flow)
        assert flownode.is_invertible() is True
    
    def testFlowNode_pretrained_node(self):
        x = numx_rand.random([100,10])
        pretrained_node = mdp.nodes.PCANode(output_dim=6)
        pretrained_node.train(x)
        pretrained_node.stop_training()
        flow = mdp.Flow([pretrained_node,
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.PCANode(output_dim=3)])
        flownode = mh.FlowNode(flow)
        while flownode.get_remaining_train_phase() > 0:
            flownode.train(x)
            flownode.stop_training()
        flownode.execute(x)
        
    def testFlowNode_pretrained_flow(self):
        flow = mdp.Flow([mdp.nodes.PolynomialExpansionNode(degree=2), 
                         mdp.nodes.PCANode(output_dim=15, reduce=True),
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.PCANode(output_dim=3, reduce=True)])
        flownode = mh.FlowNode(flow)
        x = numx_rand.random([300,20])
        while flownode.get_remaining_train_phase() > 0:
            flownode.train(x)
            flownode.stop_training()
        # build new flownode with the trained nodes    
        flownode = mh.FlowNode(flow)
        assert not flownode.is_training()
        flownode.execute(x)

    def testLayer(self):
        node1 = mdp.nodes.PCANode(input_dim=10, output_dim=5)
        node2 = mdp.nodes.PCANode(input_dim=17, output_dim=3)
        node3 = mdp.nodes.PCANode(input_dim=3, output_dim=1)
        x = numx_rand.random([100,30]).astype('f')
        layer = mh.Layer([node1, node2, node3])
        layer.train(x)
        y = layer.execute(x)
        assert layer.dtype == numx.dtype('f')
        assert y.dtype == layer.dtype
        
    def testLayer_invertibility(self):
        node1 = mdp.nodes.PCANode(input_dim=10, output_dim=10)
        node2 = mdp.nodes.PCANode(input_dim=17, output_dim=17)
        node3 = mdp.nodes.PCANode(input_dim=3, output_dim=3)
        x = numx_rand.random([100,30]).astype('f')
        layer = mh.Layer([node1, node2, node3])
        layer.train(x)
        y = layer.execute(x)
        x_inverse = layer.inverse(y)
        assert numx.all(numx.absolute(x - x_inverse) < 0.001)
        
    def testLayer_invertibility2(self):
        # reduce the dimensions, so input_dim != output_dim
        node1 = mdp.nodes.PCANode(input_dim=10, output_dim=8)
        node2 = mdp.nodes.PCANode(input_dim=17, output_dim=12)
        node3 = mdp.nodes.PCANode(input_dim=3, output_dim=3)
        x = numx_rand.random([100,30]).astype('f')
        layer = mh.Layer([node1, node2, node3])
        layer.train(x)
        y = layer.execute(x)
        layer.inverse(y)
        
    def testSameInputLayer(self):
        node1 = mdp.nodes.PCANode(input_dim=10, output_dim=5)
        node2 = mdp.nodes.PCANode(input_dim=10, output_dim=3)
        node3 = mdp.nodes.PCANode(input_dim=10, output_dim=1)
        x = numx_rand.random([100,10]).astype('f')
        layer = mh.SameInputLayer([node1, node2, node3])
        layer.train(x)
        y = layer.execute(x)
        assert layer.dtype == numx.dtype('f')
        assert y.dtype == layer.dtype
        
    def testCloneLayer(self):
        node = mdp.nodes.PCANode(input_dim=10, output_dim=5)
        x = numx_rand.random([10,70]).astype('f')
        layer = mh.CloneLayer(node, 7)
        layer.train(x)
        y = layer.execute(x)
        assert layer.dtype == numx.dtype('f')
        assert y.dtype == layer.dtype
        
    def testSwitchboardInverse1(self):
        sboard = mh.Switchboard(input_dim=3,
                                connections=[2,0,1])
        assert sboard.is_invertible()
        y = numx.array([[2,3,4],[5,6,7]])
        x = sboard.inverse(y)
        assert numx.all(x == numx.array([[3,4,2],[6,7,5]]))
    
    def testSwitchboardInverse2(self):
        sboard = mh.Switchboard(input_dim=3,
                                connections=[2,1,1])
        assert not sboard.is_invertible()
        
    ## Tests for ChannelSwitchboard ##
    
    def testOutChannelInput(self):
        sboard = mh.ChannelSwitchboard(input_dim=6,
                                       connections=[5,5,
                                                    0,1],
                                       out_channel_dim=2,
                                       in_channel_dim=2)
        assert numx.all(sboard.get_out_channel_input(0) ==
                        numx.array([5,5]))
        assert numx.all(sboard.get_out_channel_input(1) ==
                        numx.array([0,1]))
        
    def testOutChannelsInputChannels(self):
        sboard = mh.ChannelSwitchboard(input_dim=6,
                                       connections=[5,5, # out chan 1
                                                    0,1], # out chan 2
                                       out_channel_dim=2,
                                       in_channel_dim=2)
        # note that there are 3 input channels
        assert numx.all(sboard.get_out_channels_input_channels(0) ==
                        numx.array([2]))
        assert numx.all(sboard.get_out_channels_input_channels(1) ==
                        numx.array([0]))
        assert numx.all(sboard.get_out_channels_input_channels([0,1]) ==
                        numx.array([0,2]))
        
    ## Tests for Rectangular2dSwitchboard ##

    def testRect2dRouting1(self):
        sboard = mh.Rectangular2dSwitchboard(x_in_channels=3, 
                                             y_in_channels=2,
                                             in_channel_dim=2,
                                             x_field_channels=2, 
                                             y_field_channels=1,
                                             x_field_spacing=1, 
                                             y_field_spacing=1)
        assert numx.all(sboard.connections == 
                               numx.array([0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 
                                           8, 9, 8, 9, 10, 11]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        # test generated switchboard
        channel_sboard = sboard.get_out_channel_node(0)
        channel_sboard.execute(x)

    def testRect2dRouting2(self):
        sboard = mh.Rectangular2dSwitchboard(x_in_channels=2, 
                                             y_in_channels=4, 
                                             in_channel_dim=1,
                                             x_field_channels=1, 
                                             y_field_channels=2,
                                             x_field_spacing=1, 
                                             y_field_spacing=2)
        assert numx.all(sboard.connections == 
                        numx.array([0, 2, 1, 3, 4, 6, 5, 7]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        # test generated switchboard
        channel_sboard = sboard.get_out_channel_node(0)
        channel_sboard.execute(x)
        
    def testRect2d_get_out_channel_node(self):
        sboard = mh.Rectangular2dSwitchboard(x_in_channels=5, 
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
        layer = mh.SameInputLayer(nodes)
        layer_y = layer.execute(x)
        assert numx.all(y==layer_y)
        
    def testRect2dException1(self):
        try:
            mh.Rectangular2dSwitchboard(x_in_channels=12, 
                                        y_in_channels=8,
                                        x_field_channels=4,
                                        # this is the problematic value: 
                                        y_field_channels=3,
                                        x_field_spacing=2, 
                                        y_field_spacing=2,
                                        in_channel_dim=3,
                                        ignore_cover=False)
        except mh.Rectangular2dSwitchboardException:
            pass
        else:
            assert False, 'Did not raise correct exception.'
            
    def testRect2dException2(self):
        try:
            mh.Rectangular2dSwitchboard(x_in_channels=12, 
                                        y_in_channels=8,
                                        x_field_channels=4,
                                        # this is the problematic value: 
                                        y_field_channels=9,
                                        x_field_spacing=2, 
                                        y_field_spacing=2,
                                        in_channel_dim=3,
                                        ignore_cover=False)
        except mh.Rectangular2dSwitchboardException:
            pass
        else:
            assert False, 'Did not raise correct exception.'
            
    def testRect2dException3(self):
        try:
            mh.Rectangular2dSwitchboard(x_in_channels=12, 
                                        y_in_channels=8,
                                        x_field_channels=4,
                                        # this is the problematic value: 
                                        y_field_channels=9,
                                        x_field_spacing=2, 
                                        y_field_spacing=2,
                                        in_channel_dim=3,
                                        ignore_cover=True)
        except mh.Rectangular2dSwitchboardException:
            pass
        else:
            assert False, 'Did not raise correct exception.'
            
    ## Tests for DoubleRect2dSwitchboard ##
    
    def testRectDoubleRouting1(self):
        sboard = mh.DoubleRect2dSwitchboard(x_in_channels=4, 
                                            y_in_channels=4, 
                                            in_channel_dim=1,
                                            x_field_channels=2, 
                                            y_field_channels=2)
        assert numx.all(sboard.connections == 
                        numx.array([0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15,
                                    # uneven fields
                                    5,6,9,10]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
            
    def testRectDoubleRouting2(self):
        sboard = mh.DoubleRect2dSwitchboard(x_in_channels=6, 
                                            y_in_channels=4, 
                                            in_channel_dim=1,
                                            x_field_channels=2, 
                                            y_field_channels=2)
        assert numx.all(sboard.connections == 
                        numx.array([0,1,6,7, 2,3,8,9, 4,5,10,11, 12,13,18,19,
                                    14,15,20,21, 16,17,22,23,
                                    # uneven fields
                                    7,8,13,14, 9,10,15,16]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        
    def testRectDoubleRouting3(self):
        sboard = mh.DoubleRect2dSwitchboard(x_in_channels=4, 
                                            y_in_channels=6, 
                                            in_channel_dim=1,
                                            x_field_channels=2, 
                                            y_field_channels=2)
        assert numx.all(sboard.connections == 
                        numx.array([0,1,4,5, 2,3,6,7, 8,9,12,13, 10,11,14,15,
                                    16,17,20,21, 18,19,22,23,
                                    # uneven fields
                                    5,6,9,10, 13,14,17,18]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        
    ## Tests for DoubleRhomb2dSwitchboard ##
        
    def testDoubleRhombRouting1(self):
        sboard = mh.DoubleRhomb2dSwitchboard(x_long_in_channels=3, 
                                             y_long_in_channels=2, 
                                             diag_field_channels=2,
                                             in_channel_dim=1)
        assert numx.all(sboard.connections == 
                        numx.array([1,6,7,4]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        
    def testDoubleRhombRouting2(self):
        sboard = mh.DoubleRhomb2dSwitchboard(x_long_in_channels=2, 
                                             y_long_in_channels=3, 
                                             diag_field_channels=2,
                                             in_channel_dim=1)
        assert numx.all(sboard.connections == 
                        numx.array([6,2,3,7]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        
    def testDoubleRhombRouting3(self):
        sboard = mh.DoubleRhomb2dSwitchboard(x_long_in_channels=4, 
                                             y_long_in_channels=2, 
                                             diag_field_channels=2,
                                             in_channel_dim=1)
        assert numx.all(sboard.connections == 
                        numx.array([1,8,9,5, 2,9,10,6]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)

    def testDoubleRhombRouting4(self):
        sboard = mh.DoubleRhomb2dSwitchboard(x_long_in_channels=2, 
                                             y_long_in_channels=4, 
                                             diag_field_channels=2,
                                             in_channel_dim=1)
        assert numx.all(sboard.connections == 
                        numx.array([8,2,3,9, 9,4,5,10]))
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
        
    def testDoubleRhombRouting5(self):
        sboard = mh.DoubleRhomb2dSwitchboard(x_long_in_channels=7, 
                                             y_long_in_channels=4, 
                                             diag_field_channels=4,
                                             in_channel_dim=1)
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)

    def testDoubleRhombRouting6(self):
        sboard = mh.DoubleRhomb2dSwitchboard(x_long_in_channels=4, 
                                             y_long_in_channels=7, 
                                             diag_field_channels=4,
                                             in_channel_dim=1)
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)

    def testDoubleRhombRouting7(self):
        sboard = mh.DoubleRhomb2dSwitchboard(x_long_in_channels=6, 
                                             y_long_in_channels=7, 
                                             diag_field_channels=4,
                                             in_channel_dim=1)
        x = numx.array([range(0, sboard.input_dim), 
                        range(101, 101+sboard.input_dim)])
        sboard.execute(x)
    
    def testHinetSimpleNet(self):
        switchboard = mh.Rectangular2dSwitchboard(x_in_channels=12, 
                                                  y_in_channels=8,
                                                  x_field_channels=4, 
                                                  y_field_channels=4,
                                                  x_field_spacing=2, 
                                                  y_field_spacing=2,
                                                  in_channel_dim=3)
        
        node = mdp.nodes.PCANode(input_dim=4*4*3, output_dim=5)
        flownode = mh.FlowNode(mdp.Flow([node,]))
        layer = mh.CloneLayer(flownode, switchboard.output_channels)
        flow = mdp.Flow([switchboard, layer])
        x = numx_rand.random([5, switchboard.input_dim])
        flow.train(x)

    def testSFANet(self):
        noisenode = mdp.nodes.NoiseNode(input_dim=20*20, 
                                        noise_args=(0, 0.0001))
        sfa_node = mdp.nodes.SFANode(input_dim=20*20, output_dim=10, dtype='f')
        switchboard = mh.Rectangular2dSwitchboard(x_in_channels=100, 
                                                  y_in_channels=100,
                                                  x_field_channels=20, 
                                                  y_field_channels=20,
                                                  x_field_spacing=10, 
                                                  y_field_spacing=10)
        flownode = mh.FlowNode(mdp.Flow([noisenode, sfa_node]))
        sfa_layer = mh.CloneLayer(flownode, switchboard.output_channels)
        flow = mdp.Flow([switchboard, sfa_layer])
        train_gen = numx.cast['f'](numx_rand.random((3, 10, 100*100)))
        flow.train([None, train_gen])
        
    def testHiNetHTML(self):
        # create some flow for testing
        noisenode = mdp.nodes.NoiseNode(input_dim=20*20, 
                                        noise_args=(0, 0.0001))
        sfa_node = mdp.nodes.SFANode(input_dim=20*20, output_dim=10)
        switchboard = mh.Rectangular2dSwitchboard(x_in_channels=100, 
                                                  y_in_channels=100,
                                                  x_field_channels=20, 
                                                  y_field_channels=20,
                                                  x_field_spacing=10, 
                                                  y_field_spacing=10)
        flownode = mh.FlowNode(mdp.Flow([noisenode, sfa_node]))
        sfa_layer = mh.CloneLayer(flownode, switchboard.output_channels)
        flow = mdp.Flow([switchboard, sfa_layer])
        # create dummy file to write the HTML representation
        html_file = StringIO.StringIO()
        hinet_html = mdp.hinet.HiNetHTMLTranslator()
        hinet_html.write_flow_to_file(flow, html_file)
        html_file.close()
        
    def testHiNetXHTML(self):
        # create some flow for testing
        sfa_node = mdp.nodes.SFANode(input_dim=20*20, output_dim=10)
        flow = mdp.Flow([sfa_node])
        # create dummy file to write the HTML representation
        html_file = StringIO.StringIO()
        hinet_html = mdp.hinet.HiNetXHTMLTranslator()
        hinet_html.write_flow_to_file(flow, html_file)
        html_file.close()
    

def get_suite(testname=None):
    return HinetTestSuite(testname=testname)

if __name__ == '__main__':
    numx_rand.seed(1268049219)
    unittest.TextTestRunner(verbosity=2).run(get_suite())
