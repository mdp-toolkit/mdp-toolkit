"""
Module for parallel flow training and execution.

Not that this module depends on bihinet, since it uses a BiFlowNode to
encapsulate the BiFlow in the tasks.
"""
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import next
from builtins import range

import itertools

import mdp
n = mdp.numx

import mdp.parallel as parallel

from bimdp import (
    BiFlow, BiFlowException, MessageResultContainer, BiCheckpointFlow,
    EXIT_TARGET
)
from bimdp.hinet import BiFlowNode


### Train Task Classes ###

class BiFlowTrainCallable(parallel.FlowTrainCallable):
    """Task implementing a single training phase in a flow for a data block."""

    def __call__(self, data):
        """Do the training and return the purged BiFlowNode.

        data -- tuple containing x and msg
        """
        x, msg = data
        while True:
            result = self._flownode.train(x, msg)
            if (result is None) or isinstance(result, dict):
                break
            elif (isinstance(result, tuple) and
                (result[2] in [1, -1, EXIT_TARGET])):
                break
            else:
                err = ("Target node not found in flow during " +
                       "training, last result: " + str(result))
                raise BiFlowException(err)
        self._flownode.bi_reset()
        if self._purge_nodes:
            parallel._purge_flownode(self._flownode)
        return self._flownode

    def fork(self):
        return self.__class__(self._flownode.fork(),
                              purge_nodes=self._purge_nodes)


### Execute Task Classes ###

class BiFlowExecuteCallable(parallel.FlowExecuteCallable):
    """Task implementing data execution for a BiFlowNode."""
    
    def __init__(self, flownode, purge_nodes=True):
        """Store everything for the execution.

        flownode -- FlowNode for the execution
        purge_nodes -- If True nodes not needed for the join will be replaced
            with dummy nodes to reduce the footprint.
        """
        super(BiFlowExecuteCallable, self).__init__(flownode,
                                                    purge_nodes=purge_nodes)

    def __call__(self, data):
        """Return the execution result and the BiFlowNode as a tuple.

        If use_fork_execute is True for the flownode then it is returned
        in the result tuple.
        """
        x, msg, target = data
        # by using _flow we do not have to reenter (like for train)
        result = self._flownode._flow.execute(x, msg, target)
        self._flownode.bi_reset()
        if self._flownode.use_execute_fork():
            if self._purge_nodes:
                parallel._purge_flownode(self._flownode)
            return (result, self._flownode)
        else:
            return (result, None)

    def fork(self):
        return self.__class__(self._flownode.fork(),
                              purge_nodes=self._purge_nodes)


### ParallelBiFlow Class ###

class ParallelBiFlowException(parallel.ParallelFlowException):
    """Standard exception for problems with ParallelBiFlow."""


class ParallelBiFlow(BiFlow, parallel.ParallelFlow):
    """A parallel  provides the tasks for parallel training.

    Note that even though a node input x or output y can be None, the data
    iterables cannot be None themselves, since they define the iterator length
    for the message iterator as well. They can, however, return None for each
    iteration step.
    """

    def __init__(self, flow, verbose=False, **kwargs):
        """Initialize the internal variables."""
        self._train_msg_iterables = None
        self._train_msg_iterator = None
        self._stop_messages = None
        self._exec_msg_iterator = None
        self._exec_target_iterator = None
        super(ParallelBiFlow, self).__init__(flow, verbose=verbose, **kwargs)

    @mdp.with_extension("parallel")
    def train(self, data_iterables, msg_iterables=None,
              stop_messages=None,
              scheduler=None,
              train_callable_class=None,
              overwrite_result_container=True,
              **kwargs):
        """Parallel version of the standard train method.

        If a scheduler is provided the training will be done in parallel on the
        scheduler.

        data_iterables -- A list of iterables, one for each node in the flow.
            The iterators returned by the iterables must
            return data arrays that are then used for the node training.
            See Flow.train for more details.
            If a custom train_callable_class is used to preprocess the data
            then other data types can be used as well.
        msg_iterables - A list of iterables for the messages.
        stop_messages -- Sequence of messages for stop_training.
        scheduler -- Value can be either None for normal training (default
            value) or a Scheduler instance for parallel training with the
            scheduler.
            If the scheduler value is an iterable or iterator then it is
            assumed that it contains a scheduler for each training phase.
            After a node has been trained the scheduler is shutdown. Note that
            you can e.g. use a generator to create the schedulers just in time.
            For nodes which are not trained the scheduler can be None.
        train_callable_class -- Class used to create training callables for the
            scheduler. By specifying your own class you can implement data
            transformations before the data is actually fed into the flow
            (e.g. from 8 bit image to 64 bit double precision).
            Note that the train_callable_class is only used if a scheduler was
            provided. If a scheduler is provided the default class used is
            NodeResultContainer.
        overwrite_result_container -- If set to True (default value) then
            the result container in the scheduler will be overwritten with an
            instance of NodeResultContainer, if it is not already an instance
            of NodeResultContainer.
        """
        if self.is_parallel_training:
            raise ParallelBiFlowException("Parallel training is underway.")
        if scheduler is None:
            if train_callable_class is not None:
                err = ("A train_callable_class was specified but no scheduler "
                       "was given, so the train_callable_class has no effect.")
                raise ParallelBiFlowException(err)
            super(ParallelBiFlow, self).train(data_iterables, msg_iterables,
                                              stop_messages,
                                              **kwargs)
        else:
            if train_callable_class is None:
                train_callable_class = BiFlowTrainCallable
            schedulers = None
            # do parallel training
            try:
                self.setup_parallel_training(
                                        data_iterables=data_iterables,
                                        msg_iterables=msg_iterables,
                                        stop_messages=stop_messages,
                                        train_callable_class=train_callable_class,
                                        **kwargs)
                # prepare scheduler
                if not isinstance(scheduler, parallel.Scheduler):
                    # scheduler contains an iterable with the schedulers
                    # self._i_train_node was set in setup_parallel_training
                    schedulers = iter(scheduler)
                    scheduler = next(schedulers)
                    if self._i_train_node > 0:
                        # dispose schedulers for pretrained nodes
                        for _ in range(self._i_train_node):
                            if scheduler is not None:
                                scheduler.shutdown()
                            scheduler = next(schedulers)
                    elif self._i_train_node is None:
                        # all nodes are already trained, dispose schedulers
                        for _ in range(len(self.flow) - 1):
                            if scheduler is not None:
                                scheduler.shutdown()
                            # the last scheduler will be shutdown in finally
                            scheduler = next(schedulers)
                    last_trained_node = self._i_train_node
                else:
                    schedulers = None
                # check that the scheduler is compatible
                if ((scheduler is not None) and
                    overwrite_result_container and
                    (not isinstance(scheduler.result_container,
                                    parallel.TrainResultContainer))):
                    scheduler.result_container = \
                        parallel.TrainResultContainer()
                while self.is_parallel_training:
                    while self.task_available:
                        task = self.get_task()
                        scheduler.add_task(*task)
                    results = scheduler.get_results()
                    if results == []:
                        err = ("Could not get any training tasks or results "
                               "for the current training phase.")
                        raise Exception(err)
                    else:
                        self.use_results(results)
                    # check if we have to switch to next scheduler
                    if ((schedulers is not None) and
                        (self._i_train_node > last_trained_node)):
                        # dispose unused schedulers
                        for _ in range(self._i_train_node - last_trained_node):
                            if scheduler is not None:
                                scheduler.shutdown()
                            scheduler = next(schedulers)
                        last_trained_node = self._i_train_node
                        # check that the scheduler is compatible
                        if ((scheduler is not None) and
                            overwrite_result_container and
                            (not isinstance(scheduler.result_container,
                                            parallel.TrainResultContainer))):
                            scheduler.result_container = \
                                            parallel.TrainResultContainer()
            finally:
                # reset remaining iterator references, which cannot be pickled
                self._train_data_iterator = None
                self._train_msg_iterator = None
                if (schedulers is not None) and (scheduler is not None):
                    scheduler.shutdown()

    def setup_parallel_training(self, data_iterables, msg_iterables=None,
                                stop_messages=None,
                                train_callable_class=BiFlowTrainCallable):
        """Prepare the flow for handing out tasks to do the training.

        After calling setup_parallel_training one has to pick up the
        tasks with get_task, run them and finally return the results via
        use_results. tasks are available as long as task_available is
        True. Training may require multiple phases, which are each closed by
        calling use_results.

        data_iterables -- A list of iterables, one for each node in the flow.
            The iterators returned by the iterables must
            return data arrays that are then used for the node training.
            See Flow.train for more details.
            If a custom train_callable_class is used to preprocess the data
            then other data types can be used as well.
        msg_iterables - A list of iterables for the messages. Can also be
            a single message if data_iterables is a single array.
        stop_messages -- Sequence of messages for stop_training.
        train_callable_class -- Class used to create training callables for the
            scheduler. By specifying your own class you can implement data
            transformations before the data is actually fed into the flow
            (e.g. from 8 bit image to 64 bit double precision).
            Note that the train_callable_class is only used if a scheduler was
            provided. If a scheduler is provided the default class used is
            NodeResultContainer.
        """
        self._bi_reset()  # normally not required, just for safety
        if self.is_parallel_training:
            err = "Parallel training is already underway."
            raise ParallelBiFlowException(err)
        self._train_callable_class = train_callable_class
        data_iterables, msg_iterables = self._sanitize_training_iterables(
                                            data_iterables=data_iterables,
                                            msg_iterables=msg_iterables)
        self._train_data_iterables = data_iterables
        self._train_msg_iterables = msg_iterables
        if stop_messages is None:
            stop_messages = [None] * len(data_iterables)
        self._stop_messages = stop_messages
        self._flownode = BiFlowNode(BiFlow(self.flow))
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
        # find next node that can be forked, if required do local training
        while self._i_train_node < len(self.flow):
            current_node = self.flow[self._i_train_node]
            if not current_node.is_training():
                self._i_train_node += 1
                continue
            iterable = self._train_data_iterables[self._i_train_node]
            msg_iterable = self._train_msg_iterables[self._i_train_node]
            iterable, msg_iterable, _ = self._sanitize_iterables(iterable,
                                                                 msg_iterable)
            try:
                self._flownode.fork()
                # fork successful, prepare parallel training
                if self.verbose:
                    print ("start parallel training phase of " +
                           "node no. %d in parallel flow" %
                           (self._i_train_node+1))
                self._train_data_iterator = iter(iterable)
                self._train_msg_iterator = iter(msg_iterable)
                first_task = self._create_train_task()
                # make sure that iterator is not empty
                if first_task is None:
                    if current_node.get_current_train_phase() == 1:
                        err_str = ("The training data iteration for node "
                                   "no. %d could not be repeated for the "
                                   "second training phase, you probably "
                                   "provided an iterator instead of an "
                                   "iterable." % (self._i_train_node+1))
                        raise mdp.FlowException(err_str)
                    else:
                        err_str = ("The training data iterator for node "
                                   "no. %d is empty." % (self._i_train_node+1))
                        raise mdp.FlowException(err_str)
                task_data_chunk = first_task[0]
                if task_data_chunk is None:
                    err = "Training data iterator is empty."
                    raise ParallelBiFlowException(err)
                # Only first task contains the new callable (enable caching).
                # A fork is not required here, since the callable is always
                # forked in the scheduler.
                self._next_task = (task_data_chunk,
                            self._train_callable_class(self._flownode,
                                                       purge_nodes=True))
                break
            except parallel.NotForkableParallelException as exception:
                if self.verbose:
                    print ("could not fork node no. %d: %s" %
                           (self._i_train_node + 1, str(exception)))
                    print ("start nonparallel training phase of " +
                           "node no. %d in parallel flow" %
                           (self._i_train_node+1))
                self._local_train_phase(iterable, msg_iterable)
                if self.verbose:
                    print ("finished nonparallel training phase of " +
                           "node no. %d in parallel flow" %
                           (self._i_train_node+1))
                if not self.flow[self._i_train_node].is_training():
                    self._i_train_node += 1
        else:
            # training is finished
            self._i_train_node = None

    def _local_train_phase(self, iterable, msg_iterable):
        """Perform a single training phase locally.

        The internal _train_callable_class is used for the training.
        """
        task_callable = self._train_callable_class(self._flownode,
                                                   purge_nodes=False)
        i_task = 0
        for (x, msg) in zip(iterable, msg_iterable):
            i_task += 1
            # Note: if x contains additional args assume that the
            # callable can handle this
            task_callable((x, msg))
            if self.verbose:
                print ("    finished nonparallel task no. %d" % i_task)
        # perform stop_training with result check
        self._stop_training_hook()
        result = self._flownode.stop_training(
                                self._stop_messages[self._i_train_node])
        self._post_stop_training_hook()
        if (result is not None) and (not isinstance(result, dict)):
            if (isinstance(result, tuple) and
                (result[2] in [1, -1, EXIT_TARGET])):
                pass
            else:
                err = ("Target node not found in flow during " +
                       "stop_training phase, last result: " +
                       str(result))
                raise BiFlowException(err)
        self._bi_reset()

    def _create_train_task(self):
        """Create and return a single training task without callable.

        Returns None if data iterator end is reached.
        Raises NoTaskException if any other problem arises.
        """
        try:
            x = next(self._train_data_iterator)
            msg = next(self._train_msg_iterator)
            return ((x, msg), None)
        except StopIteration:
            return None

    @mdp.with_extension("parallel")  # needed for fork in local scheduler
    def execute(self, iterable=None, msg_iterable=None, target_iterable=None,
                scheduler=None,
                execute_callable_class=None,
                overwrite_result_container=True):
        """Execute the flow and return (y, msg).

        If a scheduler is provided the execution will be done in parallel on
        the scheduler.

        iterable -- Single array or iterable.
        msg_iterable -- Single message or iterable.
        target_iterable -- Single target or iterable.
        scheduler -- Value can be either None for normal execution (default
            value) or a Scheduler instance for parallel execution with the
            scheduler.
        execute_callable_class -- Class used to create execution callables for
            the scheduler. By specifying your own class you can implement data
            transformations before the data is actually fed into the flow
            (e.g. from 8 bit image to 64 bit double precision).
            Note that the execute_callable_class is only used if a scheduler was
            provided. If a scheduler is provided the default class used is
            NodeResultContainer.
        overwrite_result_container -- If set to True (default value) then
            the result container in the scheduler will be overwritten with an
            instance of OrderedResultContainer, if it is not already an
            instance of OrderedResultContainer.
        """
        if self.is_parallel_training:
            raise ParallelBiFlowException("Parallel training is underway.")
        if scheduler is None:
            if execute_callable_class is not None:
                err = ("A execute_callable_class was specified but no "
                       "scheduler was given, so the execute_callable_class "
                       "has no effect.")
                raise ParallelBiFlowException(err)
            return super(ParallelBiFlow, self).execute(iterable, msg_iterable,
                                                       target_iterable)
        if execute_callable_class is None:
            execute_callable_class = BiFlowExecuteCallable
        # check that the scheduler is compatible
        if overwrite_result_container:
            if not isinstance(scheduler.result_container,
                              parallel.ExecuteResultContainer):
                scheduler.result_container = parallel.ExecuteResultContainer()
        # do parallel execution
        self._flownode = BiFlowNode(BiFlow(self.flow))
        try:
            self.setup_parallel_execution(
                                iterable=iterable,
                                msg_iterable=msg_iterable,
                                target_iterable=target_iterable,
                                execute_callable_class=execute_callable_class)
            while self.task_available:
                task = self.get_task()
                scheduler.add_task(*task)
            result = self.use_results(scheduler.get_results())
        finally:
            # reset remaining iterator references, which cannot be pickled
            self._exec_data_iterator = None
            self._exec_msg_iterator = None
            self._exec_target_iterator = None
        return result

    def setup_parallel_execution(self, iterable, msg_iterable=None,
                                 target_iterable=None,
                                 execute_callable_class=BiFlowExecuteCallable):
        """Prepare the flow for handing out tasks to do the execution.

        Instead of automatically executing the _flow with the iterable, it only
        prepares the tasks for the scheduler.

        iterable -- Single array or iterable.
        msg_iterable -- Single message or iterable.
        target_iterable -- Single target or iterable.
        execute_callable_class -- Class used to create execution callables for
            the scheduler. By specifying your own class you can implement data
            transformations before the data is actually fed into the flow
            (e.g. from 8 bit image to 64 bit double precision).
            Note that the execute_callable_class is only used if a scheduler
            was provided. If a scheduler is provided the default class used is
            NodeResultContainer.
        """
        self._bi_reset()  # normally not required, just for safety
        if self.is_parallel_training:
            raise ParallelBiFlowException("Parallel training is underway.")
        self._execute_callable_class = execute_callable_class
        iterable, msg_iterable, target_iterable = self._sanitize_iterables(
                                                           iterable,
                                                           msg_iterable,
                                                           target_iterable)
        self._exec_data_iterator = iter(iterable)
        self._exec_msg_iterator = iter(msg_iterable)
        self._exec_target_iterator = iter(target_iterable)
        first_task = self._create_execute_task()
        if first_task is None:
            err = ("The execute data iterable is empty.")
            raise mdp.FlowException(err)
        task_data_chunk = first_task[0]
        if task_data_chunk is None:
            err = "Execution data iterable is empty."
            raise ParallelBiFlowException(err)
        # Only first task contains the new callable (enable caching).
        # A fork is not required here, since the callable is always
        # forked in the scheduler.
        self._next_task = (task_data_chunk,
                           self._execute_callable_class(self._flownode,
                                                        purge_nodes=True))

    def _create_execute_task(self):
        """Create and return a single execution task.

        Returns None if data iterator end is reached.
        Raises NoTaskException if no task is available.
        """
        try:
            x = next(self._exec_data_iterator)
            msg = next(self._exec_msg_iterator)
            target = next(self._exec_target_iterator)
            return ((x, msg, target), None)
        except StopIteration:
            return None

    def use_results(self, results):
        """Use the result from the scheduler.

        During parallel training this will start the next training phase.
        For parallel execution this will return the result, like a normal
        execute would. In addition it will join any forked nodes.

        results -- Iterable containing the results, normally the return value
            of scheduler.ResultContainer.get_results().
            The individual results can be the return values of the tasks.
        """
        if self.is_parallel_training:
            for result in results:
                self._flownode.join(result)
            # perform local stop_training with result check
            self._stop_training_hook()
            result = self._flownode.stop_training(
                                    self._stop_messages[self._i_train_node])
            self._post_stop_training_hook()
            if (result is not None):
                target = result[2]
                # values of +1, -1 and EXIT_TARGET are tolerated
                if target not in [1, -1, EXIT_TARGET]:
                    err = ("Target node not found in flow during " +
                           "stop_training phase, last result: " +
                           str(result))
                    raise BiFlowException(err)
            self._flownode.bi_reset()
            if self.verbose:
                print ("finished parallel training phase of node no. " +
                       "%d in parallel flow" % (self._i_train_node+1))
            if not self.flow[self._i_train_node].is_training():
                self._i_train_node += 1
            self._next_train_phase()
        elif self.is_parallel_executing:
            self._exec_data_iterator = None
            self._exec_msg_iterator = None
            self._exec_target_iterator = None
            y_results = []
            msg_results = MessageResultContainer()
            # use internal flownode to join all biflownodes
            self._flownode = BiFlowNode(BiFlow(self.flow))
            for result_tuple in results:
                result, forked_biflownode = result_tuple
                # consolidate results
                if isinstance(result, tuple) and (len(result) == 2):
                    y, msg = result
                    msg_results.add_message(msg)
                else:
                    y = result
                if y is not None:
                    try:
                        y_results.append(y)
                    except:
                        err = "Some but not all y return values were None."
                        raise BiFlowException(err)
                else:
                    y_results = None
                # join biflownode
                if forked_biflownode is not None:
                    self._flownode.join(forked_biflownode)
            # return results
            if y_results is not None:
                y_results = n.concatenate(y_results)
            return (y_results, msg_results.get_message())
        else:
            err = "It seems that there are no results to retrieve."
            raise BiFlowException(err)


class ParallelCheckpointBiFlow(mdp.parallel.ParallelCheckpointFlow,
                               ParallelBiFlow, BiCheckpointFlow):
    """Parallel version of CheckpointFlow.

    Can be used for saving intermediate results.
    """

    def train(self, data_iterables, checkpoints, msg_iterables=None,
              stop_messages=None,
              scheduler=None,
              train_callable_class=None,
              overwrite_result_container=True,
              **kwargs):
        """Train all trainable nodes in the flow.

        Same as the train method in ParallelFlow, but with additional support
        of checkpoint functions as in CheckpointFlow.
        """
        # this call goes via ParallelCheckpointFlow to ParallelBiFlow and then:
        #     the train call in ParallelBiFlow then goes to BiCheckpointFlow
        #     the setup_parallel_training goes to ParallelCheckpointBiFlow
        kwargs["checkpoints"] = checkpoints
        super(ParallelCheckpointBiFlow, self).train(
                        data_iterables=data_iterables,
                        scheduler=scheduler,
                        train_callable_class=train_callable_class,
                        overwrite_result_container=overwrite_result_container,
                        msg_iterables=msg_iterables,
                        **kwargs)

    def setup_parallel_training(self, data_iterables, checkpoints,
                                msg_iterables=None,
                                train_callable_class=BiFlowTrainCallable,
                                **kwargs):
        """Checkpoint version of parallel training."""
        # this call goes to ParallelCheckpointFlow and then ParallelBiFlow
        super(ParallelCheckpointBiFlow, self).setup_parallel_training(
                                    data_iterables=data_iterables,
                                    checkpoints=checkpoints,
                                    train_callable_class=train_callable_class,
                                    msg_iterables=msg_iterables,
                                    **kwargs)
