
import unittest
import numpy as n

import mdp

import binet


class TestParallelBiFlow(unittest.TestCase):
    
    def test_nonparallel_flow(self):
        """Test a ParallelBiFlow with normal non-parallel nodes."""
        flow = binet.ParallelBiFlow([mdp.nodes.SFANode(output_dim=5),
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

    def test_parallel_multiphase(self):
        """Test training and execution with multiple training phases.
        
        The node with multiple training phases is a hinet.FlowNode. 
        """
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flownode = binet.ParallelBiFlowNode(binet.BiFlow([sfa_node, 
                                                          sfa2_node]))
        flow = binet.ParallelBiFlow([flownode,
                         mdp.nodes.PolynomialExpansionNode(degree=2),
                         mdp.nodes.SFANode(output_dim=5)])
        data_iterables = [[n.random.random((30,10)) for _ in range(6)], 
                          None, 
                          [n.random.random((30,10)) for _ in range(6)]]
        scheduler = mdp.parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        x = n.random.random([100,10])
        flow.execute([x for _ in range(4)], scheduler=scheduler)
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
        flow = binet.ParallelBiFlow([sfa_node, sfa2_node])
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
        

def get_suite():
    suite = unittest.TestSuite()
    # suite.addTest(unittest.makeSuite(TestBetaResultContainer))
    suite.addTest(unittest.makeSuite(TestParallelBiFlow))
    return suite
            
if __name__ == '__main__':
    unittest.main() 