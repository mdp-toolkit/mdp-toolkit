from _tools import *

import mdp.parallel as parallel
n = numx

# TODO: add tests for new use_execute_fork mechanism

def test_tasks():
    """Test parallel training and execution by running the tasks."""
    flow = parallel.ParallelFlow([
                        mdp.nodes.SFANode(output_dim=5),
                        mdp.nodes.PolynomialExpansionNode(degree=3),
                        mdp.nodes.SFANode(output_dim=20)])
    data_iterables = [[n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)],
                      None,
                      [n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)]]
    scheduler = parallel.Scheduler()
    flow.train(data_iterables, scheduler=scheduler)
    # parallel execution
    iterable = [n.random.random((20,10)) for _ in xrange(6)]
    flow.execute(iterable, scheduler=scheduler)

def test_non_iterator():
    """Test parallel training and execution with a single array."""
    flow = parallel.ParallelFlow([
                        mdp.nodes.SFANode(output_dim=5),
                        mdp.nodes.PolynomialExpansionNode(degree=3),
                        mdp.nodes.SFANode(output_dim=20)])
    data_iterables = n.random.random((200,10))*n.arange(1,11)
    scheduler = parallel.Scheduler()
    flow.train(data_iterables, scheduler=scheduler)
    # test execution
    x = n.random.random((100,10))
    flow.execute(x)

def test_multiple_schedulers():
    """Test parallel flow training with multiple schedulers."""
    flow = parallel.ParallelFlow([
                        mdp.nodes.SFANode(output_dim=5),
                        mdp.nodes.PolynomialExpansionNode(degree=3),
                        mdp.nodes.SFANode(output_dim=20)])
    data_iterables = [[n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)],
                      None,
                      [n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)]]
    schedulers = [parallel.Scheduler(), None, parallel.Scheduler()]
    flow.train(data_iterables, scheduler=schedulers)
    # parallel execution
    iterable = [n.random.random((20,10)) for _ in xrange(6)]
    flow.execute(iterable, scheduler=parallel.Scheduler())

def test_multiple_schedulers2():
    """Test parallel flow training with multiple schedulers (part 2)."""
    # now the first node is untrainable as well
    flow = parallel.ParallelFlow([
                        mdp.nodes.PolynomialExpansionNode(degree=2),
                        mdp.nodes.SFANode(output_dim=5),
                        mdp.nodes.PolynomialExpansionNode(degree=3),
                        mdp.nodes.SFANode(output_dim=20)])
    data_iterables = [None,
                      [n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)],
                      None,
                      [n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)]]
    schedulers = [None, parallel.Scheduler(), None, parallel.Scheduler()]
    flow.train(data_iterables, scheduler=schedulers)
    # parallel execution
    iterable = [n.random.random((20,10)) for _ in xrange(6)]
    flow.execute(iterable, scheduler=parallel.Scheduler())

def test_multiphase():
    """Test parallel training and execution for nodes with multiple
    training phases.
    """
    sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
    sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
    flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
    flow = parallel.ParallelFlow([
                        flownode,
                        mdp.nodes.PolynomialExpansionNode(degree=2),
                        mdp.nodes.SFANode(output_dim=5)])
    data_iterables = [[n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)],
                      None,
                      [n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)]]
    scheduler = parallel.Scheduler()
    flow.train(data_iterables, scheduler=scheduler)
    # test normal execution
    x = n.random.random([100,10])
    flow.execute(x)
    # parallel execution
    iterable = [n.random.random((20,10)) for _ in xrange(6)]
    flow.execute(iterable, scheduler=scheduler)

def test_firstnode():
    """Test special case in which the first node is untrainable.

    This tests the proper initialization of the internal variables.
    """
    flow = parallel.ParallelFlow([
                        mdp.nodes.PolynomialExpansionNode(degree=2),
                        mdp.nodes.SFANode(output_dim=20)])
    data_iterables = [None,
                       n.random.random((6,20,10))]
    scheduler = parallel.Scheduler()
    flow.train(data_iterables, scheduler=scheduler)

def test_multiphase_checkpoints():
    """Test parallel checkpoint flow."""
    sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
    sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
    flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
    flow = parallel.ParallelCheckpointFlow([
                        flownode,
                        mdp.nodes.PolynomialExpansionNode(degree=2),
                        mdp.nodes.SFANode(output_dim=5)])
    data_iterables = [[n.random.random((30,10)) for _ in xrange(6)],
                      None,
                      [n.random.random((30,10)) for _ in xrange(6)]]
    checkpoint = mdp.CheckpointFunction()
    scheduler = parallel.Scheduler()
    flow.train(data_iterables, scheduler=scheduler, checkpoints=checkpoint)

def test_nonparallel1():
    """Test training for mixture of parallel and non-parallel nodes."""
    sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
    # TODO: use a node with no parallel here
    sfa2_node = mdp.nodes.CuBICANode(input_dim=8)
    flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
    flow = parallel.ParallelFlow([
                        flownode,
                        mdp.nodes.PolynomialExpansionNode(degree=2),
                        mdp.nodes.SFANode(output_dim=5)])
    data_iterables = [[n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)],
                      None,
                      [n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)]]
    scheduler = parallel.Scheduler()
    flow.train(data_iterables, scheduler=scheduler)
    # test execution
    x = n.random.random([100,10])
    flow.execute(x)

def test_nonparallel2():
    """Test training for mixture of parallel and non-parallel nodes."""
    # TODO: use a node with no parallel here
    sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
    sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
    flownode = mdp.hinet.FlowNode(mdp.Flow([sfa_node, sfa2_node]))
    flow = parallel.ParallelFlow([
                        flownode,
                        mdp.nodes.PolynomialExpansionNode(degree=2),
                        mdp.nodes.SFANode(output_dim=5)])
    data_iterables = [[n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)],
                      None,
                      [n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)]]
    scheduler = parallel.Scheduler()
    flow.train(data_iterables, scheduler=scheduler)
    # test execution
    x = n.random.random([100,10])
    flow.execute(x)

def test_nonparallel3():
    """Test training for non-parallel nodes."""
    # TODO: use a node with no parallel here
    sfa_node = mdp.nodes.SFANode(input_dim=10, output_dim=8)
    # TODO: use a node with no parallel here
    sfa2_node = mdp.nodes.SFA2Node(input_dim=8, output_dim=6)
    flow = parallel.ParallelFlow([sfa_node, sfa2_node])
    data_iterables = [[n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)],
                      [n.random.random((30,10))*n.arange(1,11)
                       for _ in xrange(6)]]
    scheduler = parallel.Scheduler()
    flow.train(data_iterables, scheduler=scheduler)
    while flow.is_parallel_training:
        results = []
        while flow.task_available():
            task = flow.get_task()
            results.append(task())
        flow.use_results(results)
    # test execution
    x = n.random.random([100,10])
    flow.execute(x)

