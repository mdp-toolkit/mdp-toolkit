
import unittest
import numpy as n

import mdp

from bimdp import BiFlow, MSG_ID_SEP
from bimdp.parallel import (
    ParallelBiFlow, ParallelBiFlowNode, ParallelCloneBiLayer
)
from bimdp.nodes import SFABiNode


class TestCloneBiLayer(unittest.TestCase):
    """Test the behavior of the BiCloneLayer."""
    
    def test_use_copies_msg(self):
        """Test the correct reaction to an outgoing use_copies message."""
        stop_result = ({"clonelayer" + MSG_ID_SEP + "use_copies": True},
                       "clonelayer")
        stop_sfa_node = SFABiNode(stop_result=stop_result,
                                  input_dim=10, output_dim=3)
        biflownode = ParallelBiFlowNode(BiFlow([stop_sfa_node]))
        clonelayer = ParallelCloneBiLayer(node=biflownode, 
                                                n_nodes=3, 
                                                use_copies=False, 
                                                node_id="clonelayer")
        data = [[n.random.random((100,30)) for _ in range(5)]]
        biflow = ParallelBiFlow([clonelayer])
        biflow.train(data, scheduler=mdp.parallel.Scheduler())
        assert(clonelayer.use_copies is True)
        

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCloneBiLayer))
    return suite
            
if __name__ == '__main__':
    unittest.main() 