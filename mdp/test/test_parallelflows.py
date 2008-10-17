
import unittest

import mdp
import mdp.parallel as parallel
from mdp import numx as n


# TODO: add test for proper copying

class TestParallelFlows(unittest.TestCase):

    def test_tasks(self):
        """Test parallel training and execution by running the tasks."""
        flow = parallel.ParallelFlow([
                            parallel.ParallelSFANode(output_dim=5),
                            mdp.nodes.PolynomialExpansionNode(degree=3),
                            parallel.ParallelSFANode(output_dim=20)])
        data_iterables = [n.random.random((6,20,10)), 
                          None, 
                          n.random.random((6,20,10))]
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        # test execution
        x = n.random.random([100,10])
        flow.execute(x)
        # parallel execution
        iterable = [n.random.random((20,10)) for _ in range(6)]
        flow.execute(iterable, scheduler=scheduler)
        
    def test_multiphase(self):
        """Test parallel training and execution for nodes with multiple
        training phases.
        """
        sfa_node = parallel.ParallelSFANode(input_dim=10, output_dim=8)
        sfa2_node = parallel.ParallelSFA2Node(input_dim=8, output_dim=6)
        flownode = parallel.ParallelFlowNode(mdp.Flow([sfa_node, sfa2_node]))
        flow = parallel.ParallelFlow([
                            flownode,
                            mdp.nodes.PolynomialExpansionNode(degree=2),
                            parallel.ParallelSFANode(output_dim=5)])
        data_iterables = [n.random.random((6,30,10)), 
                          None, 
                          n.random.random((6,30,10))]
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        # test normal execution
        x = n.random.random([100,10])
        flow.execute(x)
        # parallel execution
        iterable = [n.random.random((20,10)) for _ in range(6)]
        flow.execute(iterable, scheduler=scheduler)
    
    def test_firstnode(self):
        """Test special case in which the first node is untrainable.
        
        This tests the proper initialization of the internal variables.
        """
        flow = parallel.ParallelFlow([
                            mdp.nodes.PolynomialExpansionNode(degree=2),
                            parallel.ParallelSFANode(output_dim=20)])
        data_iterables = [None, 
                           n.random.random((6,20,10))]
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
            
    def test_multiphase_checkpoints(self):
        """Test parallel checkpoint flow."""
        sfa_node = parallel.ParallelSFANode(input_dim=10, output_dim=8)
        sfa2_node = parallel.ParallelSFA2Node(input_dim=8, output_dim=6)
        flownode = parallel.ParallelFlowNode(mdp.Flow([sfa_node, sfa2_node]))
        flow = parallel.ParallelCheckpointFlow([
                            flownode,
                            mdp.nodes.PolynomialExpansionNode(degree=2),
                            parallel.ParallelSFANode(output_dim=5)])
        data_iterables = [n.random.random((6,30,10)), 
                           None, 
                           n.random.random((6,30,10))]
        checkpoint = mdp.CheckpointFunction()
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler, checkpoints=checkpoint) 
            
    def test_nonparallel1(self):
        """Test training for mixture of parallel and non-parallel nodes."""
        sfa_node = parallel.ParallelSFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flownode = parallel.ParallelFlowNode(mdp.Flow([sfa_node, sfa2_node]))
        flow = parallel.ParallelFlow([
                            flownode,
                            mdp.nodes.PolynomialExpansionNode(degree=2),
                            parallel.ParallelSFANode(output_dim=5)])
        data_iterables = [n.random.random((6,30,10)), 
                          None, 
                          n.random.random((6,30,10))]
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        # test execution
        x = n.random.random([100,10])
        flow.execute(x)
        
    def test_nonparallel2(self):
        """Test training for mixture of parallel and non-parallel nodes."""
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = parallel.ParallelSFA2Node(input_dim=8, output_dim=6)
        flownode = parallel.ParallelFlowNode(mdp.Flow([sfa_node, sfa2_node]))
        flow = parallel.ParallelFlow([
                            flownode,
                            mdp.nodes.PolynomialExpansionNode(degree=2),
                            parallel.ParallelSFANode(output_dim=5)])
        data_iterables = [n.random.random((6,30,10)), 
                          None, 
                          n.random.random((6,30,10))]
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        # test execution
        x = n.random.random([100,10])
        flow.execute(x)
        
    def test_nonparallel3(self):
        """Test training for non-parallel nodes."""
        sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
        sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
        flow = parallel.ParallelFlow([sfa_node, sfa2_node])
        data_iterables = [n.random.random((6,30,10)), 
                           n.random.random((6,30,10))]
        scheduler = parallel.Scheduler()
        flow.train(data_iterables, scheduler=scheduler)
        while flow.is_parallel_training():
            results = []
            while flow.task_available():
                task = flow.get_task()
                results.append(task())
            flow.use_results(results)
        # test execution
        x = n.random.random([100,10])
        flow.execute(x)
        
    def test_makeparallel(self):
        node1 = mdp.nodes.PCANode(output_dim=20)
        node2 = mdp.nodes.PolynomialExpansionNode(degree=1)
        node3 = mdp.nodes.SFA2Node(output_dim=10)
        flow = mdp.Flow([node1, node2, node3])
        parallel_flow = parallel.make_flow_parallel(flow)
        flow = parallel_flow.copy()
        scheduler = parallel.Scheduler()
        train_iterables = [mdp.numx.random.random((3, 200, 50)) 
                           for _ in range(3)]
        parallel_flow.train(train_iterables, scheduler=scheduler)
        flow.train(train_iterables)
        reconstructed_flow = parallel.unmake_flow_parallel(parallel_flow)
        x = mdp.numx.random.random((3, 50))
        y1 = flow.execute(x)
        y2 = reconstructed_flow.execute(x)
        print y1,  "\n\n", y2
        self.assertTrue(n.all(y1 == y2))

    
def get_suite(testname=None):
    # this suite just ignores the testname argument
    # you can't select tests by name here!
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestParallelFlows))
    return suite
            
if __name__ == '__main__':
    unittest.main() 
