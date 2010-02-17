
import unittest
import numpy as n

import mdp

import binet
from binet import SFABiNode


class TestBiFlowNode(unittest.TestCase):
    """Test the behavior of the BiFlowNode."""
    
    def test_two_nodes1(self):
        """Test a TestBiFlowNode with two normal nodes."""
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flownode = binet.BiFlowNode(binet.BiFlow([sfa_node, sfa2_node]))
        for _ in range(2):
            for _ in range(6):
                flownode.train(n.random.random((30,10)))
            flownode.stop_training()
        x = n.random.random([100,10])
        flownode.execute(x)
    
    def test_two_nodes2(self):
        """Test a TestBiFlowNode with two normal nodes using a normal Flow."""
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flownode = binet.BiFlowNode(binet.BiFlow([sfa_node, sfa2_node]))
        flow = mdp.Flow([flownode])
        data_iterables = [[n.random.random((30,10)) for _ in range(6)]]
        flow.train(data_iterables)
        x = n.random.random([100,10])
        flow.execute(x)
        
    def test_pretrained_nodes(self):
        """Test a TestBiFlowNode with two normal pretrained nodes."""
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flownode = binet.BiFlowNode(binet.BiFlow([sfa_node, sfa2_node]))
        flow = mdp.Flow([flownode])
        data_iterables = [[n.random.random((30,10)) for _ in range(6)]]
        flow.train(data_iterables)
        pretrained_flow = flow[0]._flow
        biflownode = binet.BiFlowNode(pretrained_flow)
        x = n.random.random([100,10])
        biflownode.execute(x)
        

class TestCloneBiLayer(unittest.TestCase):
    """Test the behavior of the BiCloneLayer."""
    
    def test_use_copies_msg(self):
        """Test the correct reaction to an outgoing use_copies message."""
        stop_result = ({"clonelayer=>use_copies": True}, 1)
        stop_sfa_node = SFABiNode(stop_result=stop_result,
                                  input_dim=10, output_dim=3)
        clonelayer = binet.CloneBiLayer(node=stop_sfa_node, 
                                        n_nodes=3, 
                                        use_copies=False, 
                                        node_id="clonelayer")
        x = n.random.random((100,30))
        clonelayer.train(x)
        clonelayer.stop_training()
        assert(clonelayer.use_copies is True)
        
    def test_use_copies_msg_flownode(self):
        """Test the correct reaction to an outgoing use_copies message."""
        stop_result = ({"clonelayer=>use_copies": True}, "clonelayer")
        stop_sfa_node = SFABiNode(stop_result=stop_result,
                                  input_dim=10, output_dim=3)
        biflownode = binet.BiFlowNode(binet.BiFlow([stop_sfa_node]))
        clonelayer = binet.CloneBiLayer(node=biflownode, 
                                        n_nodes=3, 
                                        use_copies=False, 
                                        node_id="clonelayer")
        x = n.random.random((100,30))
        clonelayer.train(x)
        clonelayer.stop_training()
        assert(clonelayer.use_copies is True)


class TestBiSwitchboardNode(unittest.TestCase):
    """Test the behavior of the BiSwitchboardNode."""
    
    def test_execute_routing(self):
        """Test the standard routing for messages."""
        sboard = binet.BiSwitchboard(input_dim=3, connections=[2,0,1])
        x = n.array([[1,2,3],[4,5,6]])
        msg = {"string": "blabla",
               "list": [1,2],
               "data": x.copy()}
        y, out_msg = sboard.execute(x, msg)
        reference_y = n.array([[3,1,2],[6,4,5]])
        self.assertTrue(n.all(y == reference_y))
        self.assertTrue(out_msg["string"] == msg["string"])
        self.assertTrue(out_msg["list"] == msg["list"])
        self.assertTrue(n.all(out_msg["data"] == reference_y))
    
    def test_inverse_message_routing(self):
        """Test the inverse routing for messages."""
        sboard = binet.BiSwitchboard(input_dim=3, connections=[2,0,1])
        x = n.array([[1,2,3],[4,5,6]])
        msg = {"string": "blabla",
               "method": "inverse",
               "list": [1,2],
               "data": x,
               "target": "test"}
        y, out_msg, target = sboard.execute(None, msg)
        self.assertTrue(y is None)
        self.assertTrue(target == "test")
        reference_y = n.array([[2,3,1],[5,6,4]])
        self.assertTrue(out_msg["string"] == msg["string"])
        self.assertTrue(out_msg["list"] == msg["list"])
        self.assertTrue(n.all(out_msg["data"] == reference_y))
    

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBiFlowNode))
    suite.addTest(unittest.makeSuite(TestCloneBiLayer))
    suite.addTest(unittest.makeSuite(TestBiSwitchboardNode))
    return suite
            
if __name__ == '__main__':
    unittest.main() 