
import unittest
import numpy as n

import mdp

import binet

from tracernode import TraceJumpBiNode, IdNode


class TestMessageResultContainer(unittest.TestCase):
    """Test the behavior of the BetaResultContainer."""
    
    # TODO: implement
    
    def test_array_dict(self):
        """Test msg being a dict containing an array."""
        pass
    
    def test_list_dict(self):
        """Test msg being a dict containing a list."""
        pass


# TODO: test global messages

class TestBiFlow(unittest.TestCase):

    def test_normal_flow(self):
        """Test a BiFlow with normal nodes."""
        flow = binet.BiFlow([mdp.nodes.SFANode(output_dim=5),
                             mdp.nodes.PolynomialExpansionNode(degree=3),
                             mdp.nodes.SFANode(output_dim=20)])
        data_iterables = [[n.random.random((20,10)) for _ in range(6)], 
                          None, 
                          [n.random.random((20,10)) for _ in range(6)]]
        flow.train(data_iterables)
        x = n.random.random([100,10])
        flow.execute(x)
        
    def test_normal_multiphase(self):
        """Test training and execution with multiple training phases.
        
        The node with multiple training phases is a hinet.FlowNode. 
        """
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
        flow = binet.BiFlow([flownode,
                             mdp.nodes.PolynomialExpansionNode(degree=2),
                             mdp.nodes.SFANode(output_dim=5)])
        data_iterables = [[n.random.random((30,10)) for _ in range(6)], 
                          None, 
                          [n.random.random((30,10)) for _ in range(6)]]
        flow.train(data_iterables)
        x = n.random.random([100,10])
        flow.execute(x)
        
    # TODO: update this test after the branching removal
#    def test_bi_training(self):
#        """Test calling bi_message during training and stop_bi_train."""
#        tracelog = []
#        verbose = False
#        node1 = TraceJumpBiNode(
#                    tracelog=tracelog,
#                    node_id="node_1",
#                    train_results=[(None, "node_3")],
#                    stop_train_results=[(None, "node_3")],
#                    verbose=verbose)
#        node2 = TraceJumpBiNode(
#                    tracelog=tracelog,
#                    node_id="node_2",
#                    train_results=[(None, "node_1")],
#                    stop_train_results=[(None, "node_1")],
#                    stop_message_results=[(None, "node_1")],
#                    verbose=verbose)
#        node3 = TraceJumpBiNode(
#                    tracelog=tracelog,
#                    node_id="node_3",
#                    train_results=[(None, "node_2")],
#                    stop_train_results=[(None, "node_2")],
#                    verbose=verbose)
#        biflow = binet.BiFlow([node1, node2, node3])
#        data_iterables = [[n.random.random((1,1)) for _ in range(2)], 
#                          [n.random.random((1,1)) for _ in range(2)], 
#                          [n.random.random((1,1)) for _ in range(2)]]
#        biflow.train(data_iterables)
#        #binet.show_training(biflow, data_iterables)
#        # tracelog reference
#        reference = [
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'),
#            # training of node 1
#            ('node_1', 'train'), 
#            ('node_3', 'message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'), 
#            ('node_1', 'train'), 
#            ('node_3', 'message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'), 
#            ('node_1', 'stop_training'), 
#            ('node_3', 'stop_message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'),
#            # training of node 2
#            ('node_1', 'execute'), 
#            ('node_2', 'train'), 
#            ('node_1', 'message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'), 
#            ('node_1', 'execute'), 
#            ('node_2', 'train'), 
#            ('node_1', 'message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'), 
#            ('node_2', 'stop_training'), 
#            ('node_1', 'stop_message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'),
#            # training of node 3
#            ('node_1', 'execute'), 
#            ('node_2', 'execute'), 
#            ('node_3', 'train'), 
#            ('node_2', 'message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'), 
#            ('node_1', 'execute'), 
#            ('node_2', 'execute'), 
#            ('node_3', 'train'), 
#            ('node_2', 'message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'), 
#            ('node_3', 'stop_training'), 
#            ('node_2', 'stop_message'), 
#            ('node_1', 'stop_message'), 
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset') ]
#        self.assertEqual(tracelog, reference)
        
#    def test_bi_execution(self):
#        """Test calling bi_message during execution."""
#        tracelog = []
#        verbose = False
#        node1 = TraceJumpBiNode(
#                    tracelog=tracelog,
#                    node_id="node_1",
#                    execute_results=[(None, 1, None, "node_3")],
#                    verbose=verbose)
#        node2 = TraceJumpBiNode(
#                    tracelog=tracelog,
#                    node_id="node_2",
#                    execute_results=[None, (None, -1, None, "node_3")],
#                    verbose=verbose)
#        node3 = TraceJumpBiNode(
#                    tracelog=tracelog,
#                    node_id="node_3",
#                    verbose=verbose)
#        biflow = binet.BiFlow([node1, node2, node3])
#        biflow.execute(n.random.random((1,1)))
#        # tracelog reference
#        reference = [
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset'),
#            ('node_1', 'execute'),
#            ('node_3', 'message'),
#            ('node_2', 'message'),
#            ('node_2', 'execute'),
#            ('node_3', 'message'),
#            ('node_1', 'execute'),
#            ('node_2', 'execute'),
#            ('node_3', 'execute'),
#            ('node_1', 'bi_reset'), 
#            ('node_2', 'bi_reset'), 
#            ('node_3', 'bi_reset')]
#        self.assertEqual(tracelog, reference)
    
#    def test_mixed_execution(self):
#        """Test calling execution of a mixed flow."""
#        tracelog = []
#        verbose = False
#        node1 = TraceJumpBiNode(
#                    tracelog=tracelog,
#                    node_id="node_1",
#                    execute_results=[(None, 1, None, "node_3")],
#                    verbose=verbose)
#        node2 = IdNode()
#        node3 = TraceJumpBiNode(
#                    tracelog=tracelog,
#                    node_id="node_3",
#                    verbose=verbose)
#        biflow = binet.BiFlow([node1, node2, node3])
#        biflow.execute(n.random.random((1,1)))
#        # tracelog reference
#        reference = [
#            ('node_1', 'bi_reset'), 
#            ('node_3', 'bi_reset'),
#            ('node_1', 'execute'),
#            ('node_3', 'message'),
#            ('node_1', 'message'),
#            ('node_3', 'execute'),
#            ('node_1', 'bi_reset'), 
#            ('node_3', 'bi_reset')]
#        self.assertEqual(tracelog, reference)
        

def get_suite():
    suite = unittest.TestSuite()
    # suite.addTest(unittest.makeSuite(TestBetaResultContainer))
    suite.addTest(unittest.makeSuite(TestBiFlow))
    return suite
            
if __name__ == '__main__':
    unittest.main() 