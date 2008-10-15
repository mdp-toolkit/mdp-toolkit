"""
Module for parallel MDP flows, which handle the tasks.

Corresponding classes for task callables and ResultContainer are derived.
"""

import mdp
from mdp import numx as n

import parallelnodes
import scheduling
import parallelhinet


### Train task classes ###

class FlowTrainCallable(scheduling.TaskCallable):
    """Implements a single training phase in a flow for a data block.
    
    A ParallelFlowNode is used to simplify the forking process and to 
    encapsulate the flow.
    """

    def __init__(self, flownode):
        """Store everything for the training.
        
        keyword arguments:
        flownode -- FlowNode containing the flow to be trained.
        """
        self._flownode = flownode
    
    def __call__(self, x):
        """Do the training and return only the trained node.
        
        x -- training data block
        """
        self._flownode.train(x)
        for node in self._flownode._flow:
            if node.is_training():
                return node
            
    def copy(self):
        return FlowTrainCallable(self._flownode.copy())
            

class NodeResultContainer(scheduling.ResultContainer):
    """Container for parallel nodes.
    
    Expects parallel nodes as results and joins them to save memory.
    A list containing one node is returned, so this container can replace
    the standard list container without any changes elsewhere.
    """
    
    def __init__(self):
        self._node = None
        
    def add_result(self, result, task_index):
        if not self._node:
            self._node = result
        else:
            self._node.join(result)
            
    def get_results(self):
        node = self._node
        self._node = None
        return [node,]
    

### Execute task classes ###

class FlowExecuteCallable(scheduling.TaskCallable):
    """Implements data execution through the whole flow.
    
    Note that one could also pass the flow itself as the callable, so this 
    class is not really needed. However, it serves as the base class for more
    more complicated callables, e.g. which do some kind of preprocessing before
    executing the data with the flow. 
    """

    def __init__(self, flow):
        """Store everything for the execution.
        
        keyword arguments:
        flow -- _flow for the execution
        """
        self._flow = flow
    
    def __call__(self, x):
        """Return the execution result.
        
        x -- data chunk
        """
        return self._flow.execute(x)
    

### Standard Helper Functions ###    

def train_parallelflow(flow, data_iterators, scheduler=None, checkpoints=None,
                       train_callable_class=FlowTrainCallable):
    """Train a parallel flow via the provided scheduler.
    
    The scheduler can use the NodeResultContainer to save memory.
    If no scheduler is provided the tasks will be run locally using the
    simple default scheduler.
    
    Nodes which are not derived from ParallelNode are directly trained here.
    """
    if checkpoints:
        flow.parallel_train(data_iterators, checkpoints, 
                            train_callable_class=train_callable_class)
    else:    
        flow.parallel_train(data_iterators, 
                            train_callable_class=train_callable_class)
    while flow.is_parallel_training():
        if scheduler == None:
            scheduler = scheduling.Scheduler(
                                        result_container=NodeResultContainer(),
                                        copy_callable=True)
        if not scheduler.copy_callable:
            err = "copy_callable should be True in scheduler during training"
            raise Exception(err)
        while flow.task_available():
            task = flow.get_task()
            scheduler.add_task(*task)
        results = scheduler.get_results()
        if results == []:
            raise Exception("Could not get any training tasks or results " +
                            "for the current training phase.")
        else:
            flow.use_results(results)


def execute_parallelflow(flow, data_iterator, scheduler=None, 
                         execute_callable_class=FlowExecuteCallable):
    """Execute a parallel flow via the provided scheduler.
    
    The execution results are returned in the correct order.
    If no scheduler is provided the tasks will be run locally.
    """
    flow.parallel_execute(data_iterator, 
                          execute_callable_class=execute_callable_class)
    if scheduler == None:
        scheduler = scheduling.Scheduler()
    while flow.task_available():
        task = flow.get_task()
        scheduler.add_task(*task)
    results = scheduler.get_results()
    return flow.use_results(results)


### ParallelFlow Class ###    

class ParallelFlowException(mdp.FlowException):
    """Exception for calling execute while _flow is in parallel training."""
    pass


class NoTaskException(ParallelFlowException):
    """Exception for problems with the task creation."""
    pass


class ParallelFlow(mdp.Flow):
    """A parallel flow provides the tasks for parallel training.
    
    After calling train or execute with data iterators one has to pick up the
    tasks with get_task, run them and finally return the result list via
    use_results. tasks are available as long as task_available() returns true.
    Training may require multiple phases, which are each closed by calling
    use_results.
    
    The training is only parallelized for nodes that are derived from 
    ParallelNode, otherwise the training is done locally. The training is also
    done locally if fork() raises a TrainingPhaseNotParallelException. This can
    be used by the node to request local training without forking.
    
    Note that train phases are always closed, so e.g. CheckpointSaveFunction
    should not expect open train phases. This is necessary since otherwise
    stop_training() would be called remotely.
    """
    
    def __init__(self, flow, crash_recovery=False, verbose=False):
        """Initialize the internal variables."""
        super(ParallelFlow, self).__init__(flow, crash_recovery=crash_recovery, 
                                           verbose=verbose)
        self._train_data_iters = None  # all training data
        # Warning: This must be an iterator, not just an iterable!
        self._train_data_iter = None  # iterator for current training
        # index of currently trained node, also used as flag for training 
        # takes value None for not training
        self._i_train_node = None
        self._flownode = None  # used during training
        # iterator for execution data 
        # also signals if parallel execution is underway
        self._exec_data_iter = None  
        self._next_task = None  # buffer for next task
        self._train_callable_class = None
        self._execute_callable_class = None
        
    def train(self, *args, **kwargs):
        """Non-parallel training as in standard flow.
        
        This method checks first if parallel training is underway.
        """
        if self.is_parallel_training():
            raise ParallelFlowException("Parallel training is underway.")
        super(ParallelFlow, self).train(*args, **kwargs)
    
    def parallel_train(self, data_iterators, 
                       train_callable_class=FlowTrainCallable):
        """Parallel version of the standard train method.
        
        Instead of automatically training the _flow, it only initializes the 
        training. The training still has to be done by generating tasks with
        get_task, executing them in the scheduler and returning the results
        with return_results.
        
        iterator -- Iterator for data chunks. If an array is given instead,
            then the standard flow train is used.
        train_callable_class -- Class used to create training callables. 
            By using a different class you can implement data transformations 
            (e.g. from 8 bit image to 64 bit double precision.
        """
        if isinstance(data_iterators, n.ndarray):
            self.train(self, data_iterators)
        else:
            self._train_callable_class = train_callable_class
            self._train_data_iters = self._train_check_iterators(data_iterators)
            self._i_train_node = 0
            self._next_train_phase()
            
    def _next_train_phase(self):
        """Find the next phase or node for parallel training.
        
        When it is found the corresponding internal variables are set.
        Nodes which are not derived from ParallelNode are trained locally. 
        If a fork() fails due to a TrainingPhaseNotParallelException
        in a certain train phase, then the training is done locally as well
        (but fork() is tested again for the next phase).
        """
        self._flownode = parallelhinet.ParallelFlowNode(mdp.Flow(self.flow))
        # find next node that can be forked, if required do local training
        while self._i_train_node < len(self.flow):
            if not self.flow[self._i_train_node].is_training():
                self._i_train_node += 1
                continue
            try:
                # test if node can be forked
                if isinstance(self.flow[self._i_train_node], 
                              parallelnodes.ParallelNode):
                    self._flownode.fork()
                else:
                    raise parallelnodes.TrainingPhaseNotParallelException()
            except parallelnodes.TrainingPhaseNotParallelException:
                if self.verbose:
                    print ("start local training phase of " + 
                           "node no. %d in parallel flow" % 
                           (self._i_train_node+1))
                data_iterator = self._train_data_iters[self._i_train_node]
                for x in data_iterator:
                    if (type(x) is tuple) or (type(x) is list):
                        arg = x[1:]
                        x = x[0]
                    else:
                        arg = ()
                    self._flownode.train(x, *arg)
                if self.verbose:
                    print ("finished local training phase of " + 
                           "node no. %d in parallel flow" % 
                           (self._i_train_node+1))
                self._flownode.stop_training()
                if not self.flow[self._i_train_node].is_training():
                    self._i_train_node += 1
            else:
                # fork successful, prepare parallel training
                if self.verbose:
                    print ("start parallel training phase of " +
                           "node no. %d in parallel flow" % 
                           (self._i_train_node+1))
                self._train_data_iter = iter(
                                    self._train_data_iters[self._i_train_node])
                task_data_chunk = self._create_train_task()[0]
                if task_data_chunk is None:
                    err = "Training data iterator is empty."
                    raise ParallelFlowException(err)
                # first task contains the new callable
                self._next_task = (task_data_chunk,
                                self._train_callable_class(self._flownode.fork()))
                break
        else:
            # training is finished
            self._i_train_node = None
            self._train_data_iters = None
            
    def _create_train_task(self):
        """Create and return a single training task without callable.
        
        Returns None if data iterator end is reached.
        Raises NoTaskException if any other problem arises.
        """
        try:
            return (self._train_data_iter.next(), None)
        except StopIteration:
            return None
        else:
            raise NoTaskException("Could not create training task.")
            
    def execute(self, *args, **kwargs):
        """Non-parallel execution as in standard flow.
        
        This method checks first if parallel training is underway.
        """
        if self.is_parallel_training():
            raise ParallelFlowException("Parallel training is underway.")
        return super(ParallelFlow, self).execute(*args, **kwargs)
       
    def parallel_execute(self, iterator, 
                         execute_callable_class=FlowExecuteCallable):
        """Parallel version of the standard execute method.
        
        Instead of automatically executing the _flow with the iterator, it only
        prepares the tasks for the scheduler.
        
        iterator -- Iterator for data chunks. If an array is given instead,
            then the standard flow execution is used.
        """
        if self.is_parallel_training():
            raise ParallelFlowException("Parallel training is underway.")
        if isinstance(iterator, n.ndarray):
            return self.execute(self, iterator)
        else:
            self._execute_callable_class = execute_callable_class
            self._exec_data_iter = iterator.__iter__()
            task_data_chunk = self._create_execute_task()[0]
            if task_data_chunk is None:
                err = "Execution data iterator is empty."
                raise ParallelFlowException(err)
            # first task contains the new callable
            self._next_task = (task_data_chunk,
                            self._execute_callable_class(mdp.Flow(self.flow)))
            
        
    def _create_execute_task(self):
        """Create and return a single execution task.
        
        Returns None if data iterator end is reached.
        Raises NoTaskException if none is available.
        """
        try:
            return (self._exec_data_iter.next(), None)
        except StopIteration:
            return None
        else:
            raise NoTaskException("Could not create execution task.")
    
    def get_task(self):
        """Return a task either for either training or execution.
        
        A a one task buffer is used to make task_available work.
        tasks are available as long as need_result returns False or all the 
        training / execution is done. If no tasks are available a NoTaskException 
        is raised.
        """
        if self._next_task is not None:
            task = self._next_task
            if self._i_train_node is not None:
                self._next_task = self._create_train_task()
            elif self._exec_data_iter is not None:
                self._next_task = self._create_execute_task()
            else:
                raise NoTaskException("No data available for execution task.")
            return task
        else:
            raise NoTaskException("No task available for execution.")
    
    def is_parallel_training(self):
        """Return True if parallel training is underway."""
        if self._i_train_node != None:
            return True
        else:
            return False
    
    def is_parallel_executing(self):
        """Return True if parallel execution is underway."""
        if self._exec_data_iter == None:
            return False
        else:
            return True
    
    def task_available(self):
        """Return True if tasks are available, otherwise False.
        
        If false is returned this can indicate that results are needed to
        continue training.   
        """
        if self._next_task is not None:
            return True
        else:
            return False
    
    def use_results(self, results):
        """Use the result from the scheduler.
        
        During parallel training this will start the next training phase.
        For parallel execution this will return the result, like a normal 
        execute would.
        
        results -- Iterable containing the results, normally the return value
            of scheduler.ResultContainer.get_results().  
            The individual results can be the return values of the tasks. 
        """
        if self.is_parallel_training():
            node = self.flow[self._i_train_node]
            for result in results:
                node.join(result)
            if self.verbose:
                print ("finished parallel training phase of node no. " + 
                       "%d in parallel flow" % (self._i_train_node+1))
            node.stop_training()
            if not node.is_training():
                self._i_train_node += 1
            self._next_train_phase()
        elif self.is_parallel_executing():
            self._exec_data_iter = None
            return n.concatenate(results)
   
    
class ParallelCheckpointFlow(ParallelFlow, mdp.CheckpointFlow):
    """Parallel version of CheckpointFlow.
    
    Can be used for saving intermediate results.
    """
    
    def parallel_train(self, data_iterators, checkpoints, 
                       train_callable_class=FlowTrainCallable):
        """Checkpoint version of parallel training."""
        self._checkpoints = self._train_check_checkpoints(checkpoints)
        super(ParallelCheckpointFlow, self).parallel_train(
                                    data_iterators, 
                                    train_callable_class=train_callable_class)
    
    def use_results(self, results):
        """Checkpoint version of use_results.
        
        Calls the checkpoint functions when necessary.
        """
        if self.is_parallel_training():
            i_node = self._i_train_node
            # save this info before use_results() is called, 
            # since afterwards it is ambiguous
            checkpoint_reached = False
            if self.flow[i_node].get_remaining_train_phase() == 1:
                checkpoint_reached = True
            ParallelFlow.use_results(self, results=results)
            if checkpoint_reached:
                if ((i_node <= len(self._checkpoints)) 
                    and self._checkpoints[i_node]):
                    dict = self._checkpoints[i_node](self.flow[i_node])
                    if dict: 
                        self.__dict__.update(dict)
        elif self.is_parallel_executing():
            return ParallelFlow.use_results(self, results=results)





