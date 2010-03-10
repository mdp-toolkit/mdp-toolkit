
import unittest
import numpy as np

import mdp

from bimdp import MessageResultContainer, BiFlow, EXIT_TARGET, nodes
from testnodes import TraceJumpBiNode, IdNode


class TestMessageResultContainer(unittest.TestCase):
    """Test the behavior of the BetaResultContainer."""
    
    def test_mixed_dict(self):
        """Test msg being a dict containing an array."""
        rescont = MessageResultContainer()
        msg1 = {
            "f": 2,
            "a": np.zeros((10,3), 'int'),
            "b": "aaa",
            "c": 1,
        }
        msg2 = {
            "a": np.ones((15,3), 'int'),
            "b": "bbb",
            "c": 3,
            "d": 1,
        }
        rescont.add_message(msg1)
        rescont.add_message(msg2)
        combined_msg = rescont.get_message()
        a = np.zeros((25,3), 'int')
        a[10:] = 1
        reference_msg = {"a": a, "c": 4, "b": "aaabbb", "d": 1, "f": 2}
        self.assertTrue(np.all(reference_msg["a"] == reference_msg["a"]))
        combined_msg.pop("a")
        reference_msg.pop("a")
        self.assertTrue(combined_msg == reference_msg)
        
    def test_none_msg(self):
        """Test with one message being None."""
        rescont = MessageResultContainer()
        msgs = [None, {"a": 1}, None, {"a": 2, "b": 1}, None]
        for msg in msgs:
            rescont.add_message(msg)
        msg = rescont.get_message()
        self.assertTrue(msg == {"a": 3, "b": 1})
        
    def test_incompatible_arrays(self):
        """Test with incompatible arrays."""
        rescont = MessageResultContainer()
        msgs = [{"a":  np.zeros((10,3))}, {"a":  np.zeros((10,4))}]
        for msg in msgs:
            rescont.add_message(msg)
        self.assertRaises(ValueError, lambda: rescont.get_message())
        

class TestBiFlow(unittest.TestCase):

    def test_normal_flow(self):
        """Test a BiFlow with normal nodes."""
        flow = BiFlow([mdp.nodes.SFANode(output_dim=5),
                       mdp.nodes.PolynomialExpansionNode(degree=3),
                       mdp.nodes.SFANode(output_dim=20)])
        data_iterables = [[np.random.random((20,10)) for _ in range(6)], 
                          None, 
                          [np.random.random((20,10)) for _ in range(6)]]
        flow.train(data_iterables)
        x = np.random.random([100,10])
        flow.execute(x)
        
    def test_normal_multiphase(self):
        """Test training and execution with multiple training phases.
        
        The node with multiple training phases is a hinet.FlowNode. 
        """
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
        flow = BiFlow([flownode,
                       mdp.nodes.PolynomialExpansionNode(degree=2),
                       mdp.nodes.SFANode(output_dim=5)])
        data_iterables = [[np.random.random((30,10)) for _ in range(6)], 
                          None, 
                          [np.random.random((30,10)) for _ in range(6)]]
        flow.train(data_iterables)
        x = np.random.random([100,10])
        flow.execute(x)
        
    def test_wrong_iterableException(self):
        samples = mdp.numx_rand.random((100,10))
        labels = mdp.numx.arange(100)
        # 1.
        # proper way of passing iterables for a normal Flow
        flow = mdp.Flow([mdp.nodes.PCANode(), mdp.nodes.FDANode()])
        flow.train([[[samples]], [[samples, labels]]])
        # 2.
        # if i give the wrong number of iterables (I forget one bracket),
        # I expect a FlowException:
        try:
            flow = mdp.Flow([mdp.nodes.PCANode(), mdp.nodes.FDANode()])
            flow.train([[samples], [samples, labels]])
            err = "Flow did not raise FlowException for wrong iterable."
            raise Exception(err)
        except mdp.FlowException:
            pass

        # do the same with a BiFlow
        # 1.
        flow = BiFlow([nodes.PCABiNode(), nodes.FDABiNode()])
        flow.train([[[samples]], [[samples, labels]]])
        # throws AttributeError: 'list' object has no attribute 'ndim'
        # 2.
        try:
            flow = BiFlow([nodes.PCABiNode(), nodes.FDABiNode()])
            flow.train([[samples], [samples, labels]])
            err = "Flow did not raise FlowException for wrong iterable."
            raise Exception(err)
        except mdp.FlowException:
            pass
        
    
    def test_training_targets(self):
        """Test targeting during training and stop_training."""
        tracelog = []
        verbose = False
        node1 = TraceJumpBiNode(
                    output_dim=1,
                    tracelog=tracelog,
                    node_id="node_1",
                    train_results=[[None]],
                    stop_train_results=[[None]],
                    execute_results=[None, (None, None, "node_3")],
                    verbose=verbose)
        node2 = TraceJumpBiNode(
                    output_dim=1,
                    tracelog=tracelog,
                    node_id="node_2",
                    train_results=[[None]],
                    stop_train_results=[(None, "node_1")],
                    execute_results=[None, (None, None, "node_1")],
                    stop_message_results=[(None, "node_1")],
                    verbose=verbose)
        node3 = TraceJumpBiNode(
                    output_dim=1,
                    tracelog=tracelog,
                    node_id="node_3",
                    train_results=[[(None, {"a": 1}, "node_2"), None]],
                    stop_train_results=[({"a": 1}, "node_2")],
                    verbose=verbose)
        biflow = BiFlow([node1, node2, node3])
        data_iterables = [[np.random.random((1,1)) for _ in range(2)], 
                          [np.random.random((1,1)) for _ in range(2)], 
                          [np.random.random((1,1)) for _ in range(2)]]
        biflow.train(data_iterables)
        # bimdp.show_training(biflow, data_iterables, debug=True)
        # tracelog reference
        reference = [
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_1', 'train'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_1', 'train'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_1', 'stop_training'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_1', 'execute'),
            ('node_2', 'train'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_1', 'execute'),
            ('node_2', 'train'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_2', 'stop_training'),
            ('node_1', 'stop_message'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_1', 'execute'),
            ('node_2', 'execute'), 
            ('node_3', 'train'),
            ('node_2', 'execute'),
            ('node_1', 'execute'),
            ('node_3', 'train'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_1', 'execute'),
            ('node_2', 'execute'),
            ('node_3', 'train'),
            ('node_2', 'execute'),
            ('node_1', 'execute'),
            ('node_3', 'train'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_3', 'stop_training'),
            ('node_2', 'stop_message'),
            ('node_1', 'stop_message'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset')
        ]
        self.assertEqual(tracelog, reference)

    def test_execute_jump(self):
        """Test jumping around during execution."""
        tracelog = []
        verbose = False
        node1 = TraceJumpBiNode(
                    tracelog=tracelog,
                    node_id="node_1",
                    execute_results=[(None, None, "node_3"),
                                     (None, None, "node_2")],
                    verbose=verbose)
        node2 = TraceJumpBiNode(
                    tracelog=tracelog,
                    node_id="node_2",
                    execute_results=[(None, None, "node_1")],
                    verbose=verbose)
        node3 = TraceJumpBiNode(
                    tracelog=tracelog,
                    node_id="node_3",
                    execute_results=[(None, None, "node_1")],
                    verbose=verbose)
        biflow = BiFlow([node1, node2, node3])
        biflow.execute(None, {"a": 1})
        # bimdp.show_execution(biflow, x=None, msg={"a": 1}, debug=True)
        # tracelog reference
        reference = [
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
            ('node_1', 'execute'),
            ('node_3', 'execute'),
            ('node_1', 'execute'),
            ('node_2', 'execute'),
            ('node_1', 'execute'),
            ('node_2', 'execute'),
            ('node_3', 'execute'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset'),
        ]
        self.assertEqual(tracelog, reference)

    def test_msg_normal_node(self):
        """Test that the msg is passed over a normal node."""
        node = IdNode()
        biflow = BiFlow([node])
        msg = {"a": 1}
        result = biflow.execute(np.random.random((1,1)), msg)
        self.assertTrue(msg == result[1])

    def test_exit_target(self):
        """Test that the magic exit target works."""
        tracelog = []
        node1 = TraceJumpBiNode(
                    tracelog=tracelog,
                    execute_results=[(None, None, EXIT_TARGET)],
                    verbose=False)
        node2 = IdNode()
        biflow = BiFlow([node1, node2])
        biflow.execute(None, {"a": 1})
        # bimdp.show_execution(biflow, x=None, msg={"a": 1}, debug=True)
        reference = [
           (None, 'bi_reset'), (None, 'execute'), (None, 'bi_reset')
        ]
        self.assertEqual(tracelog, reference)
    

def get_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMessageResultContainer))
    suite.addTest(unittest.makeSuite(TestBiFlow))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
