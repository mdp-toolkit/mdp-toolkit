from builtins import range
from builtins import object
import mdp
from mdp import numx as n
from bimdp.nodes import SFABiNode, SFA2BiNode
from bimdp.parallel import ParallelBiFlow

# TODO: maybe test the helper classes as well, e.g. the new callable

class TestParallelBiNode(object):

    def test_stop_message_attribute(self):
        """Test that the stop_result attribute is present in forked node."""
        stop_result = ({"test": "blabla"}, "node123")
        x = n.random.random([100,10])
        node = SFABiNode(stop_result=stop_result)
        try:
            mdp.activate_extension("parallel")
            node2 = node.fork()
            node2.train(x)
            forked_result = node2.stop_training()
            assert forked_result == (None,) + stop_result
            # same with derived sfa2 node
            node = SFA2BiNode(stop_result=stop_result)
            mdp.activate_extension("parallel")
            node2 = node.fork()
            node2.train(x)
            forked_result = node2.stop_training()
            assert forked_result == (None,) + stop_result
        finally:
            mdp.deactivate_extension("parallel")


class TestParallelBiFlow(object):

    def test_nonparallel_flow(self):
        """Test a ParallelBiFlow with standard nodes."""
        flow = ParallelBiFlow([mdp.nodes.SFANode(output_dim=5),
                               mdp.nodes.PolynomialExpansionNode(degree=3),
                               mdp.nodes.SFANode(output_dim=20)])
        data_iterables = [[n.random.random((20,10)) for _ in range(6)],
                          None,
                          [n.random.random((20,10)) for _ in range(6)]]
        scheduler = mdp.parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        x = n.random.random([100,10])
        flow.execute(x)
        iterator = [n.random.random((20,10)) for _ in range(6)]
        flow.execute(iterator, scheduler=scheduler)
        scheduler.shutdown()

    def test_mixed_parallel_flow(self):
        """Test a ParallelBiFlow with both standard and BiNodes."""
        flow = ParallelBiFlow([mdp.nodes.PCANode(output_dim=8),
                               SFABiNode(output_dim=5),
                               SFA2BiNode(output_dim=20)])
        data_iterables = [[n.random.random((20,10)) for _ in range(6)]] * 3
        scheduler = mdp.parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        x = n.random.random([100,10])
        flow.execute(x)
        iterator = [n.random.random((20,10)) for _ in range(6)]
        flow.execute(iterator, scheduler=scheduler)
        scheduler.shutdown()

    def test_parallel_process(self):
        """Test training and execution with multiple training phases.

        The node with multiple training phases is a hinet.FlowNode.
        """
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flow = ParallelBiFlow([sfa_node, sfa2_node])
        data_iterables = [[n.random.random((30,10)) for _ in range(6)],
                          [n.random.random((30,10)) for _ in range(7)]]
        scheduler = mdp.parallel.ProcessScheduler(n_processes=2)
        flow.train(data_iterables, scheduler=scheduler)
        flow.execute(data_iterables[1], scheduler=scheduler)
        x = n.random.random([100,10])
        flow.execute(x)
        iterator = [n.random.random((20,10)) for _ in range(6)]
        flow.execute(iterator, scheduler=scheduler)
        scheduler.shutdown()
