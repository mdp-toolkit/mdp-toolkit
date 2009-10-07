
import unittest

import mdp
from mdp import numx as n
import mdp.parallel as parallel
import mdp.hinet as hinet


class TestParallelHinetNodes(unittest.TestCase):
    """Tests for ParallelFlowNode."""
    
    def setUp(self):
        if "parallel" in mdp.get_active_extensions():
            self.set_parallel = False
        else:
            mdp.activate_extension("parallel")
            self.set_parallel = True
            
    def tearDown(self):
        if self.set_parallel:
            mdp.deactivate_extension("parallel")

    def test_flownode(self):
        """Test ParallelFlowNode."""
        flow = mdp.Flow([mdp.nodes.SFANode(output_dim=5),
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.SFANode(output_dim=3)])
        flownode = mdp.hinet.FlowNode(flow)
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
        sfa_node = mdp.nodes.SFANode(input_dim=20*20, output_dim=10)
        switchboard = hinet.Rectangular2dSwitchboard(x_in_channels=100, 
                                                     y_in_channels=100, 
                                                     x_field_channels=20, 
                                                     y_field_channels=20,
                                                     x_field_spacing=10, 
                                                     y_field_spacing=10)
        flownode = mdp.hinet.FlowNode(mdp.Flow([noisenode, sfa_node]))
        sfa_layer = mdp.hinet.CloneLayer(flownode, 
                                                switchboard.out_channels)
        flow = parallel.ParallelFlow([switchboard, sfa_layer])
        data_iterables = [None,
                          [n.random.random((10, 100*100)) for _ in range(3)]]
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        
    def test_layer(self):
        """Test Simple random test with three nodes."""
        node1 = mdp.nodes.SFANode(input_dim=10, output_dim=5)
        node2 = mdp.nodes.SFANode(input_dim=17, output_dim=3)
        node3 = mdp.nodes.SFANode(input_dim=3, output_dim=1)
        layer = mdp.hinet.Layer([node1, node2, node3])
        flow = parallel.ParallelFlow([layer])
        data_iterables = [[n.random.random((10, 30)) for _ in range(3)]]
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)


def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestParallelHinetNodes))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
