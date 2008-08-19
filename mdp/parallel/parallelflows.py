"""
Module for parallel MDP flows, which handle the jobs.

Corresponding classes for Job and ResultContainer are derived. 
"""

import mdp

import parallelnodes
import resultorder
import scheduling


class FlowTrainJob(scheduling.Job):
    """Job implementing a single training phase in a flow for a data block."""

    def __init__(self, preflow, node, x):
        """Store everything for the training.
        
        keyword arguments:
        preflow -- x is executed by the preflow, the result is used for the
            training of the node
        node -- node to be trained
        x -- training data block
        """
        self._preflow = preflow
        self._node = node
        self._x = x
    
    def __call__(self):
        """Do the training and return only the trained node."""
        x = self._preflow.execute(self._x)
        self._node.train(x)
        return self._node
    
    
class FlowExecuteJob(scheduling.Job):
    """Job implementing data execution through the whole flow."""

    def __init__(self, flow, x):
        """Store everything for the execution.
        
        keyword arguments:
        flow -- _flow for the execution
        x -- data chunk
        """
        self._flow = flow
        self._x = x
    
    def __call__(self):
        """Return the execution result."""
        return self._flow.execute(self._x)
    
    
class OrderedFlowExecuteJob(resultorder.OrderedJob):
    """Ordered version of FlowExecuteJob."""

    def __init__(self, flow, marked_x):
        self._marker = marked_x[0]
        self._x = marked_x[1]
        self._flow = flow
    
    def __call__(self):
        """Return the execution result."""
        result = self._flow.execute(self._x)
        return self._apply_marker(result)
    

class NodeResultContainer(scheduling.ResultContainer):
    """Container for parallel nodes.
    
    Expects parallel nodes as results and joins them to save memory.
    A list containing one node is returned, so this container can replace
    the standard list container without any changes elsewhere.
    """
    
    def __init__(self):
        self._node = None
        
    def add_result(self, result):
        if not self._node:
            self._node = result
        else:
            self._node.join(result)
            
    def get_results(self):
        node = self._node
        self._node = None
        return [node,]


class ParallelFlowException(mdp.FlowException):
    """Exception for calling execute while _flow is in parallel training."""
    pass

class NoJobException(ParallelFlowException):
    """Exception for problems with the job creation."""
    pass


def train_parallelflow(flow, data_iterators, scheduler=None, checkpoints=None,
                       train_job_class=FlowTrainJob):
    """Train a parallel flow via the provided scheduler.
    
    The scheduler can use the NodeResultContainer to save memory.
    If no scheduler is provided the jobs will be run locally using the
    simple default scheduler.
    
    Nodes which are not derived from ParallelNode are directly trained here.
    """
    if checkpoints:
        flow.parallel_train(data_iterators, checkpoints, 
                            train_job_class=train_job_class)
    else:    
        flow.parallel_train(data_iterators, train_job_class=train_job_class)
    while flow.is_parallel_training():
        if scheduler == None:
            scheduler = scheduling.SimpleScheduler(result_container=
                                                         NodeResultContainer())
        while flow.job_available():
            job = flow.get_job()
            scheduler.add_job(job)
        results = scheduler.get_results()
        if results == []:
            raise Exception("Could not get any training jobs or results " +
                            "for the current training phase.")
        else:
            flow.use_results(results)


def execute_parallelflow(flow, data_iterator, scheduler=None, 
                         execute_job_class=FlowExecuteJob):
    """Execute a parallel flow via the provided scheduler.
    
    The execution results are returned. 
    If no scheduler is provided the jobs will be run locally.
    """
    flow.parallel_execute(data_iterator, execute_job_class=execute_job_class)
    if scheduler == None:
        scheduler = scheduling.SimpleScheduler(result_container=
                                                scheduler.ListResultContainer())
    while flow.job_available():
        job = flow.get_job()
        scheduler.add_job(job)
    results = scheduler.get_results()
    return flow.use_results(results)


class ParallelFlow(mdp.Flow):
    """A parallel flow provides the jobs for parallel training.
    
    After calling train or execute with data iterators one has to pick up the
    jobs with get_job, run them and finally return the result list via
    use_results. Jobs are available as long as job_available() returns true.
    Training may require multiple phases, which are each closed by calling
    use_results.
    
    The training can only be parallelized for nodes that are derived from
    ParallelNode, otherwise the training is done locally. 
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
        # iterator for execution data 
        # also signals if parallel execution is underway
        self._exec_data_iter = None  
        self._next_job = None  # buffer for next job
        self._train_job_class = None
        self._execute_job_class = None
        
    def train(self, *args, **kwargs):
        """Non-parallel training as in standard flow.
        
        This method checks first if parallel training is underway.
        """
        if self.is_parallel_training():
            raise ParallelFlowException("Parallel training is underway.")
        super(ParallelFlow, self).train(*args, **kwargs)
    
    def parallel_train(self, data_iterators, train_job_class=FlowTrainJob):
        """Parallel version of the standard train method.
        
        Instead of automatically training the _flow, it only initializes the 
        training. The training still has to be done by generating jobs with
        get_job, executing them in the scheduler and returning the results
        with return_results.
        
        iterator -- Iterator for data chunks. If an array is given instead,
            then the standard flow train is used.
        train_job_class -- Class used to create training jobs. By using a
            different class you can implementent data transformations (e.g.
            from 8 bit image to 64 bit double precision.
        """
        if isinstance(data_iterators, mdp.numx.ndarray):
            self.train(self, data_iterators)
        else:
            self._train_job_class = train_job_class
            self._train_data_iters = \
                                self._train_check_iterators(data_iterators)
            self._i_train_node = 0
            self._next_train_phase()
           
    def _next_train_phase(self):
        """Find the next phase or node for parallel training.
        
        When it is found the corresponding internal variables are set.
        Nodes which are not derived from ParallelNode are directly trained 
        here. If a fork() fails due to a TrainingPhaseNotParallelException
        in a certain train phase, then the training is done locally as well
        (but fork() is tested again for the next phase).
        """
        i_node = self._i_train_node  # keep these synchronized at all times
        can_fork = False  # flag which can be set in inner loop
        while i_node < len(self.flow) and not can_fork:
            if self.flow[i_node].is_training():
                if isinstance(self.flow[i_node], parallelnodes.ParallelNode):
                    while self.flow[i_node].get_remaining_train_phase() >= 1:
                        # Test if node can actually fork, otherwise perform 
                        # the single training phase locally. Go through 
                        # all train phases for this node if necessary.
                        try:
                            self.flow[i_node].fork()
                        except parallelnodes.TrainingPhaseNotParallelException:
                            if self.verbose:
                                print ("start local training phase of " + 
                                       "node no. %d in parallel flow" % 
                                       (i_node+1))
                            data_iterator = self._train_data_iters[i_node]
                            node = self.flow[i_node]
                            ## begin code taken from flow._train_node()
                            for x in data_iterator:
                                if isinstance(x, (list, tuple)):
                                    arg = x[1:]
                                    x = x[0]
                                else:
                                    arg = ()
                                if i_node > 0: 
                                    x = self._execute_seq(x, i_node-1)
                                node.train(x, *arg)
                            ## end code taken from flow._train_node()
                            if self.verbose:
                                print ("finished local training phase of " + 
                                       "node no. %d in parallel flow" % 
                                       (i_node+1))
                            if node.get_remaining_train_phase() > 1:
                                node.stop_training()
                            else:
                                self._i_train_node += 1
                                i_node = self._i_train_node
                                break
                        else:
                            # fork successful, prepare parallel training
                            if self.verbose:
                                print ("start parallel training phase of " +
                                       "node no. %d in parallel flow" % 
                                       (i_node+1))
                            self._train_data_iter = iter(
                                                self._train_data_iters[i_node])
                            self._next_job = self._create_train_job()
                            can_fork = True
                            break
                    else:
                        # training of this node is done, pick next one
                        self._i_train_node += 1
                        i_node = self._i_train_node
                else:
                    # train non-parallel nodes locally
                    if self.verbose:
                        print ("start local training of non-parallel node " + 
                               "no. %d in parallel flow" % (i_node+1))
                    self._train_node(
                        data_iterator=self._train_data_iters[i_node], 
                        nodenr=i_node)
                    if self.verbose:
                        print ("finished local training of non-parallel node " + 
                               "no. %d in parallel flow" % (i_node+1))
                    self._i_train_node += 1
                    i_node = self._i_train_node
            else:
                self._i_train_node += 1
                i_node = self._i_train_node
        if i_node >= len(self.flow):
            # training is finished
            self._i_train_node = None
            self._train_data_iters = None
            
    def _create_train_job(self):
        """Create and return a single training job.
        
        Returns None if data iterator end is reached.
        Raises NoJobException if none are available.
        """
        try:
            preflow = mdp.Flow(self.flow[:self._i_train_node])
            node = self.flow[self._i_train_node].fork()
            return self._train_job_class(preflow, node,
                                         self._train_data_iter.next())
        except StopIteration:
            return None
        else:
            raise NoJobException("Could not create training job.")
            
    def execute(self, *args, **kwargs):
        """Non-parallel execution as in standard flow.
        
        This method checks first if parallel training is underway.
        """
        if self.is_parallel_training():
            raise ParallelFlowException("Parallel training is underway.")
        return super(ParallelFlow, self).execute(*args, **kwargs)
       
    def parallel_execute(self, iterator, execute_job_class=FlowExecuteJob):
        """Parallel version of the standard execute method.
        
        Instead of automatically executing the _flow with the iterator, it only
        prepares the jobs for the scheduler.
        
        iterator -- Iterator for data chunks. If an array is given instead,
            then the standard flow execution is used.
        execute_job_class -- Similar to train_job_class, but for execution.
        """
        if self.is_parallel_training():
            raise ParallelFlowException("Parallel training is underway.")
        if isinstance(iterator, mdp.numx.ndarray):
            return mdp.Flow.execute(self, iterator)
        else:
            self._execute_job_class = execute_job_class
            self._exec_data_iter = iterator.__iter__()
            self._next_job = self._create_execute_job()
        
    def _create_execute_job(self):
        """Create and return a single execution job.
        
        Returns None if data iterator end is reached.
        Raises NoJobException if none is available.
        """
        try:
            return self._execute_job_class(mdp.Flow(self.flow), 
                                           self._exec_data_iter.next())
        except StopIteration:
            return None
        else:
            raise NoJobException("Could not create execution job.")
    
    def get_job(self):
        """Return a job either for either training or execution.
        
        A a one job buffer is used to make job_available work.
        Jobs are available as long as need_result returns False or all the 
        training / execution is done. If no jobs are available a NoJobException 
        is raised.
        """
        if self._next_job != None:
            job = self._next_job
            if self._i_train_node != None:
                self._next_job = self._create_train_job()
            elif self._exec_data_iter != None:
                self._next_job = self._create_execute_job()
            else:
                raise NoJobException("No data available for execution job.")
            return job
        else:
            raise NoJobException("No job available for execution.")
    
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
    
    def job_available(self):
        """Return True if jobs are available, otherwise False.
        
        If false is returned this can indiciate that results are needed to
        continue training.   
        """
        if self._next_job != None:
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
            The individual results can be the return values of the jobs. 
        """
        if self.is_parallel_training():
            node = self.flow[self._i_train_node]
            for result in results:
                node.join(result)
            if self.verbose:
                print ("finished parallel training phase of node no. " + 
                       "%d in parallel flow" % (self._i_train_node+1))
            if node.get_remaining_train_phase() > 1:
                    node.stop_training()
            else:
                self._i_train_node += 1
            self._next_train_phase()
        elif self.is_parallel_executing():
            self._exec_data_iter = None
            return mdp.numx.concatenate(results)
   
    
class ParallelCheckpointFlow(ParallelFlow, mdp.CheckpointFlow):
    """Parallel version of CheckpointFlow.
    
    Can be used for saving intermediate results.
    """
    
    def parallel_train(self, data_iterators, checkpoints, 
                       train_job_class=FlowTrainJob):
        """Checkpoint version of parallel training."""
        self._checkpoints = self._train_check_checkpoints(checkpoints)
        ParallelFlow.parallel_train(self, data_iterators, 
                                    train_job_class=train_job_class)
    
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





