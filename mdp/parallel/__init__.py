"""
This is the MDP package for parallel processing.

It is designed to work with nodes for which a large part of the computations
are embaressingly parallel (like in PCANode). The hinet package is also fully
supported, i.e., there are parallel versions of all hinet nodes.

This package consists of two decoupled parts. The first part consists of
parallel versions of the familiar MDP structures (nodes and flows). At the top
there is the ParallelFlow, which generates jobs that can be executed in 
parallel. 
The second part consists of the schedulers. They take jobs and execute them in 
a more or less parallel way (e.g. in multiple processes). So they are designed
to deal with the more technical aspects of the parallelization, but do not
have to know anything about flows or nodes.
"""


from scheduling import (Job, TestJob, ResultContainer, ListResultContainer,
                        Scheduler, SimpleScheduler)
from resultorder import (OrderedListResultContainer, OrderedJob, 
                         OrderedIterable)
from process_schedule import ProcessScheduler
from parallelnodes import (ParallelNode, TrainingPhaseNotParallelException,
                           ParallelPCANode, ParallelWhiteningNode,
                           ParallelSFANode, ParallelSFA2Node)
from parallelflows import (FlowTrainJob, FlowExecuteJob, OrderedFlowExecuteJob,
                           NodeResultContainer, 
                           ParallelFlowException, NoJobException, 
                           train_parallelflow, execute_parallelflow,
                           ParallelFlow, ParallelCheckpointFlow)
from parallelhinet import (ParallelFlowNode, ParallelLayer, ParallelCloneLayer)

del scheduling
del resultorder
del process_schedule
del parallelnodes
del parallelflows
del parallelhinet
