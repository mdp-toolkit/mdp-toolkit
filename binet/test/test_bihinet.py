
import unittest
import numpy as n

import mdp

import binet


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
        

class BiSFANode(binet.BiNode, mdp.nodes.SFANode):
    pass


class TestCloneBiLayer(unittest.TestCase):
    """Test the behavior of the BiCloneLayer."""
    
    def test_use_copies_msg(self):
        """Test the correct reaction to an outgoing use_copies message."""
        stop_msg = {"clonelayer=>use_copies": True}
        stop_sfa_node = BiSFANode(stop_msg=stop_msg,
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
        stop_msg = {"clonelayer=>use_copies": True}
        stop_sfa_node = BiSFANode(stop_msg=stop_msg,
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
    
    def test_message_routing(self):
        """Test the standard feed-forward routing."""
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
    
    def test_down_routing(self):
        """Test the top-down routing."""
        sboard = binet.BiSwitchboard(input_dim=3, connections=[2,0,1])
        x = n.array([[1,2,3],[4,5,6]])
        msg = {"string": "blabla",
               "send_down": True,
               "list": [1,2],
               "data": x,
               "target": "test"}
        out_msg, target = sboard.message(msg)
        self.assertTrue(target == "test")
        reference_y = n.array([[2,3,1],[5,6,4]])
        self.assertTrue(out_msg["string"] == msg["string"])
        self.assertTrue(out_msg["list"] == msg["list"])
        self.assertTrue(n.all(out_msg["data"] == reference_y))
    
    def test_updata_params(self):
        """Test changing switching direction and target via msg."""
        sboard = binet.BiSwitchboard(input_dim=3, connections=[2,0,1])
        x = n.array([[1,2,3],[4,5,6]])
        msg = {"string": "blabla",
               "list": [1,2],
               "data": x,
               "send_down": True,
               "target": "test2"}
        out_msg, target = sboard.message(msg)
        self.assertTrue(target == "test2")
        reference_y = n.array([[2,3,1],[5,6,4]])
        self.assertTrue(out_msg["string"] == msg["string"])
        self.assertTrue(out_msg["list"] == msg["list"])
        self.assertTrue(n.all(out_msg["data"] == reference_y))
        

class TestBiMeanSwitchboard(unittest.TestCase):
    
    def test_rec_down_execute(self):
        """Test correct downward execution of a MeanBiSwitchboard."""
        sboard = binet.MeanBiSwitchboard(input_dim=3,
                                         connections=[0,2,1,1,0])
        y = n.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        x = sboard._down_execute(y)
        ref_x = n.array([[(1.0 + 5.0) / 2,
                          (3.0 + 4.0) / 2,
                          2.0]])
        self.assertTrue(n.all(x == ref_x))
        
    def test_rec_down_execute2(self):
        """Test correct downward execution for larger y."""
        sboard = binet.MeanBiSwitchboard(input_dim=3,
                                         connections=[0,2,1,1,0])
        y = n.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                     [1.5, 5.0, 1.2, 3.7, 1.1]])
        x = sboard._down_execute(y)
        ref_x = n.array([[(1.0 + 5.0) / 2, (3.0 + 4.0) / 2, 2.0],
                         [(1.5 + 1.1) / 2, (1.2 + 3.7) / 2, 5.0]])
        self.assertTrue(n.all(x == ref_x))
    

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBiFlowNode))
    suite.addTest(unittest.makeSuite(TestCloneBiLayer))
    suite.addTest(unittest.makeSuite(TestBiSwitchboardNode))
    suite.addTest(unittest.makeSuite(TestBiMeanSwitchboard))
    return suite
            
if __name__ == '__main__':
    unittest.main() 