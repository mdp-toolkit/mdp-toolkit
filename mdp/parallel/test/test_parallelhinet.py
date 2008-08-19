
import unittest

import mdp
import mdp.parallel as parallel
import mdp.hinet as hinet
n = mdp.numx


class TestParallelFlowNode(unittest.TestCase):
    """Tests for ParallelFlowNode."""

    def test_flownode(self):
        """Test ParallelFlowNode."""
        flow = mdp.Flow([parallel.ParallelSFANode(output_dim=5),
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         parallel.ParallelSFANode(output_dim=3)])
        flownode = parallel.ParallelFlowNode(flow)
        x = n.random.random([100,50])
        chunksize = 25
        chunks = [x[i*chunksize : (i+1)*chunksize] 
                    for i in range(len(x)/chunksize)]
        while flownode.get_remaining_train_phase() > 0:
            for chunk in chunks:
                forked_node = flownode.fork()
                forked_node.train(chunk)
                flownode.join(forked_node)
            flownode.stop_training()
        # test execution
        flownode.execute(x)
        
    def test_parallelnet(self):
        """Test a simple parallel net with big data. 
        
        Includes ParallelFlowNode, ParallelCloneLayer, ParallelSFANode
        and training via a ParallelFlow.
        """
        noisenode = mdp.nodes.NormalNoiseNode(input_dim=20*20, 
                                              noise_args=(0,0.0001))
        sfa_node = parallel.ParallelSFANode(input_dim=20*20, output_dim=10)
        switchboard = hinet.Rectangular2dSwitchboard(x_in_channels=100, 
                                                     y_in_channels=100, 
                                                     x_field_channels=20, 
                                                     y_field_channels=20,
                                                     x_field_spacing=10, 
                                                     y_field_spacing=10)
        flownode = parallel.ParallelFlowNode(mdp.Flow([noisenode, sfa_node]))
        sfa_layer = parallel.ParallelCloneLayer(flownode, 
                                                switchboard.output_channels)
        flow = parallel.ParallelFlow([switchboard, sfa_layer])
        train_gen = n.random.random((3, 10, 100*100))
        parallel.train_parallelflow(flow, [None, train_gen])
        
        
class TestParallelLayer(unittest.TestCase):
    """Tests for TestParallelLayer."""

    def test_layer(self):
        """Simple random test with two nodes."""
        node1 = parallel.ParallelSFANode(input_dim=10, output_dim=5)
        node2 = parallel.ParallelSFANode(input_dim=17, output_dim=3)
        node3 = parallel.ParallelSFANode(input_dim=3, output_dim=1)
        layer = parallel.ParallelLayer([node1, node2, node3])
        flow = parallel.ParallelFlow([layer])
        train_gen = n.random.random((3, 10, 30))
        parallel.train_parallelflow(flow, [train_gen])


def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestParallelFlowNode))
    suite.addTest(unittest.makeSuite(TestParallelLayer))
    return suite
            
if __name__ == '__main__':
    unittest.main() 