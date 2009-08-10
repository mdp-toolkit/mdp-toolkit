"""
This is the MDP package for parallel processing.

It is designed to work with nodes for which a large part of the computation
is embaressingly parallel (like in PCANode). The hinet package is also fully
supported, i.e., there are parallel versions of all hinet nodes.

This package consists of two decoupled parts. The first part consists of
parallel versions of the familiar MDP structures (nodes and flows). At the top
there is the ParallelFlow, which generates tasks that are processed in 
parallel (this can be done automatically in the train or execute methods).

The second part consists of the schedulers. They take tasks and process them in 
a more or less parallel way (e.g. in multiple processes). So they are designed
to deal with the more technical aspects of the parallelization, but do not
have to know anything about flows or nodes.
"""


from scheduling import (ResultContainer, ListResultContainer,
                        OrderedResultContainer, TaskCallable, SqrTestCallable,
                        SleepSqrTestCallable, TaskCallableWrapper, Scheduler)
from process_schedule import ProcessScheduler
import parallelnodes
from parallelnodes import (ParallelExtensionNode,
                           TrainingPhaseNotParallelException)
from parallelflows import (FlowTaskCallable, FlowTrainCallable,
                           FlowExecuteCallable, NodeResultContainer,
                           ParallelFlowException, NoTaskException,
                           ParallelFlow, ParallelCheckpointFlow)
import parallelhinet

try:
    import pp
    import pp_support
except ImportError:
    pass

del scheduling
del process_schedule
del parallelflows

# Note: the modules with the actual extension node classes are still available 

__all__ = ["ResultContainer", "ListResultContainer", "OrderedResultContainer",
           "TaskCallable", "SqrTestCallable", "SleepSqrTestCallable",
           "TaskCallableWrapper", "Scheduler",
           "ProcessScheduler",
           "ParallelExtensionNode", "TrainingPhaseNotParallelException",
           "FlowTrainCallable", "NodeResultContainer", "FlowExecuteCallable",
           "ParallelFlowException", "NoTaskException",
           "ParallelFlow", "ParallelCheckpointFlow"]
