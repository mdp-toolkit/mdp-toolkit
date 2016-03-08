from builtins import range
from builtins import object
import py.test

import mdp
from mdp import numx as np

from bimdp import (
    MessageResultContainer, BiFlow, BiFlowException, EXIT_TARGET, nodes,
)
from ._tools import TraceJumpBiNode, IdNode


class TestMessageResultContainer(object):
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
        assert np.all(reference_msg["a"] == reference_msg["a"])
        combined_msg.pop("a")
        reference_msg.pop("a")
        assert combined_msg == reference_msg

    def test_none_msg(self):
        """Test with one message being None."""
        rescont = MessageResultContainer()
        msgs = [None, {"a": 1}, None, {"a": 2, "b": 1}, None]
        for msg in msgs:
            rescont.add_message(msg)
        msg = rescont.get_message()
        assert msg == {"a": 3, "b": 1}

    def test_incompatible_arrays(self):
        """Test with incompatible arrays."""
        rescont = MessageResultContainer()
        msgs = [{"a":  np.zeros((10,3))}, {"a":  np.zeros((10,4))}]
        for msg in msgs:
            rescont.add_message(msg)
        py.test.raises(ValueError, rescont.get_message)


class TestBiFlow(object):

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

    def test_index_with_node_ids(self):
        """Test a BiFlow indexed by keys."""
        pca_node = nodes.PCABiNode(node_id="pca")
        biflow = BiFlow([pca_node])
        x = biflow["pca"]
        assert x is pca_node
        assert pca_node in biflow
        assert 'pca' in biflow

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

    def test_fda_binode(self):
        """Test using the FDABiNode in a BiFlow."""
        samples = mdp.numx_rand.random((100,10))
        labels = mdp.numx.arange(100)
        flow = BiFlow([mdp.nodes.PCANode(), nodes.FDABiNode()])
        flow.train([[samples],[samples]], [None,[{"labels": labels}]])

    def test_wrong_argument_handling(self):
        """Test correct error for additional arguments in Node instance."""
        samples = mdp.numx_rand.random((100,10))
        labels = mdp.numx.arange(100)
        # labels argument of FDANode is not supported in biflow
        flow = BiFlow([mdp.nodes.PCANode(), mdp.nodes.FDANode()])
        # the iterables are passed as if this were a normal Flow
        py.test.raises(BiFlowException,
                       flow.train, [[samples], [samples, labels]])
        # messing up the data iterables further doesn't matter, this is
        # actually interpreted as three data chunks for the FDANode training,
        # since argument iterables are not supported by BiFlow
        py.test.raises(BiFlowException,
                       flow.train, [[samples], [samples, labels, labels]])

    def test_training_targets(self):
        """Test targeting during training and stop_training."""
        tracelog = []
        verbose = False
        node1 = TraceJumpBiNode(
                    output_dim=1,
                    tracelog=tracelog,
                    node_id="node_1",
                    train_results=[[None]],
                    stop_train_results=[None],
                    execute_results=[None, (None, {"b": 2}, "node_3"),
                                     (None, {"b": 2}, EXIT_TARGET),],
                    verbose=verbose)
        node2 = TraceJumpBiNode(
                    output_dim=1,
                    tracelog=tracelog,
                    node_id="node_2",
                    train_results=[[None]],
                    stop_train_results=[(None, {"b": 2}, "node_1")],
                    execute_results=[None, (None, None, "node_1"),
                                     (None, None, "node_1")],
                    verbose=verbose)
        node3 = TraceJumpBiNode(
                    output_dim=1,
                    tracelog=tracelog,
                    node_id="node_3",
                    train_results=[[(None, {"a": 1}, "node_2"), None]],
                    stop_train_results=[(None, {"a": 1}, "node_2")],
                    execute_results=[(None, {"b": 2}, EXIT_TARGET)],
                    verbose=verbose)
        biflow = BiFlow([node1, node2, node3])
        data_iterables = [[np.random.random((1,1)) for _ in range(2)],
                          [np.random.random((1,1)) for _ in range(2)],
                          [np.random.random((1,1)) for _ in range(2)]]
        biflow.train(data_iterables)
        # print ",\n".join(str(log) for log in tracelog)
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
            ('node_1', 'execute'),
            ('node_2', 'execute'),
            ('node_3', 'execute'),
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
            ('node_2', 'execute'),
            ('node_3', 'execute'),
            ('node_1', 'bi_reset'),
            ('node_2', 'bi_reset'),
            ('node_3', 'bi_reset')
        ]
        assert tracelog == reference

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
        assert tracelog == reference

    def test_msg_normal_node(self):
        """Test that the msg is passed over a normal node."""
        node = IdNode()
        biflow = BiFlow([node])
        msg = {"a": 1}
        result = biflow.execute(np.random.random((1,1)), msg)
        assert msg == result[1]

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
        assert tracelog == reference

    def test_append_node_copy(self):
        """Test that appending a node does not perform a deept copy."""
        node1 = nodes.IdentityBiNode()
        node2 = nodes.IdentityBiNode()
        flow = BiFlow([node1])
        flow += node2
        assert flow[0] is node1
        assert type(flow) is BiFlow


