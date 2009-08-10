
import unittest
import numpy as n

import mdp

import binet


class BiSFANode(binet.BiNode, mdp.nodes.SFANode):
    pass


class TestCloneBiLayer(unittest.TestCase):
    """Test the behavior of the BiCloneLayer."""
    
    def test_use_copies_msg(self):
        """Test the correct reaction to an outgoing use_copies message."""
        stop_msg = {"clonelayer=>use_copies": True}
        stop_sfa_node = BiSFANode(stop_msg=stop_msg,
                                  input_dim=10, output_dim=3)
        biflownode = binet.ParallelBiFlowNode(binet.BiFlow([stop_sfa_node]))
        clonelayer = binet.ParallelCloneBiLayer(node=biflownode, 
                                                n_nodes=3, 
                                                use_copies=False, 
                                                node_id="clonelayer")
        data = [[n.random.random((100,30)) for _ in range(5)]]
        biflow = binet.ParallelBiFlow([clonelayer])
        #biflow.train(data, scheduler=mdp.parallel.Scheduler())
        biflow.train(data, scheduler=mdp.parallel.Scheduler())
        assert(clonelayer.use_copies is True)
        

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCloneBiLayer))
    return suite
            
if __name__ == '__main__':
    unittest.main() 