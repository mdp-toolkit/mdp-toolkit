from builtins import str
import mdp
from mdp import numx
from .linear_flows import FlowException, FlowExceptionCR, _sys, _traceback
from collections import deque as _deque


class OnlineFlowException(mdp.MDPException):
    """Base class for exceptions in Flow subclasses."""
    pass


class OnlineFlow(mdp.Flow):
    """An 'OnlineFlow' is a sequence of nodes that are trained online and executed
    together to form a more complex algorithm.  Input data is sent to the
    first node and is successively processed by the subsequent nodes along
    the sequence.

    Using an online flow as opposed to manually handling a set of nodes has a
    clear advantage: The general online flow implementation automatates the
    training (including supervised training and multiple training phases),
    execution, and inverse execution (if defined) of the whole sequence.

    To understand the compatible node sequences for an OnlineFlow, the following terminology is useful:
       A "trainable" node: node.is_trainable() returns True, node.is_training() returns True.
       A "trained" node: node.is_trainable() returns True, node.is_training() returns False.
       A "non-trainable" node: node.is_trainable() returns False, node.is_training() returns False.

    OnlineFlow node sequence can contain
    (a) only OnlineNodes
        (Eg. [OnlineCenteringNode(), IncSFANode()],
    or
    (b) a mix of OnlineNodes and trained/non-trainable Nodes
        (eg. [a fully trained PCANode, IncSFANode()] or [QuadraticExpansionNode(), IncSFANode()],
    or
    (c) a mix of OnlineNodes/trained/non-trainable Nodes and a terminal trainable Node (but not an OnlineNode) whose
    training hasn't finished
        (eg. [IncSFANode(), QuadraticExpansionNode(), a partially or untrained SFANode]).

    Differences between a Flow and an OnlineFlow:
    a) In Flow, data is processed sequentially, training one node at a time. That is, the second
       node's training starts only after the first node is "trained". Whereas, in an OnlineFlow data is
       processed simultaneously training all the nodes at the same time.
       Eg:

       flow = Flow([node1, node2]), onlineflow = OnlineFlow([node1, node2])

       Let input x = [x_0, x_1, ...., x_n], where x_t a sample or a mini batch of samples.

       Flow training:
            node1 trains on the entire x. While node1 is training, node2 is inactive.
            node1 training completes. node2 training begins on the node1(x).

            Therefore, Flow goes through all the data twice. Once for each node.

       OnlineFlow training:
            node1 trains on x_0. node2 trains on the output of node1 (node1(x_0))
            node1 trains on x_1. node2 trains on the output of node1 (node1(x_1))
            ....
            node1 trains on x_n. node2 trains on the output of node1 (node1(x_n))

            OnlineFlow goes through all the data only once.

    b) Flow requires a list of dataiterables with a length equal to the
       number of nodes or a single numpy array. OnlineFlow requires only one
       input dataiterable as each node is trained simultaneously.

    c) Additional train args (supervised labels etc) are passed to each node through the
       node specific dataiterable. OnlineFlow requires the dataiterable to return a list
       that contains tuples of args for each node: [x, (node0 args), (node1 args), ...]. See
       train docstring.

    Crash recovery is optionally available: in case of failure the current
    state of the flow is saved for later inspection.

    OnlineFlow objects are Python containers. Most of the builtin 'list'
    methods are available. An 'OnlineFlow' can be saved or copied using the
    corresponding 'save' and 'copy' methods.

    """

    def __init__(self, flow, crash_recovery=False, verbose=False):
        super(OnlineFlow, self).__init__(flow, crash_recovery, verbose)
        # check if the list of nodes is compatible. Compatible node sequences:
        # (a) A sequence of OnlineNodes, or
        # (b) a mix of OnlineNodes and trained/non-trainable Nodes, or
        # (c) a mix of OnlineNodes/trained/non-trainable Nodes and a terminal trainable Node whose
        # training hasn't finished.
        self._check_compatibility(flow)
        # collect train_args for each node
        self._train_arg_keys_list, self._train_args_needed_list = self._get_required_train_args_from_flow(flow)

    def _train_node(self, data_iterable, nodenr):
        err_str = ('Not used in %s' % str(type(self).__name__))
        OnlineFlowException(err_str)

    def _get_required_train_args_from_flow(self, flow):
        _train_arg_keys_list = [self._get_required_train_args(node) for node in flow]
        _train_args_needed_list = [bool(len(train_arg_keys)) for train_arg_keys in _train_arg_keys_list]
        return _train_arg_keys_list, _train_args_needed_list

    def _train_nodes(self, data_iterables):
        empty_iterator = True
        for x in data_iterables:
            if (type(x) is tuple) or (type(x) is list):
                args = x[1:]
                x = x[0]
                if len(args) != len(self.flow):
                    err = ("Wrong number of argument-tuples provided by " +
                           "the iterable (%d needed, %d given).\n" % (len(self.flow), len(args)))
                    raise OnlineFlowException(err)
            else:
                args = ()
            empty_iterator = False
            for nodenr in xrange(len(self.flow)):
                try:
                    node = self.flow[nodenr]
                    # check if the required number of arguments was given
                    if self._train_args_needed_list[nodenr]:
                        # the nodenr'th arguments tuple are passed only to the
                        # currently training node, allowing the implementation of
                        # supervised nodes
                        arg = args[nodenr]

                        if len(self._train_arg_keys_list[nodenr]) != len(arg):
                            err = ("Wrong number of arguments provided by " +
                                   "the iterable for node #%d " % nodenr +
                                   "(%d needed, %d given).\n" %
                                   (len(self._train_arg_keys_list[nodenr]), len(arg)) +
                                   "List of required argument keys: " +
                                   str(self._train_arg_keys_list[nodenr]))
                            raise OnlineFlowException(err)
                        if node.is_training():
                            node.train(x, *arg)
                    else:
                        if node.is_training():
                            node.train(x)

                    # input for the next node
                    x = node.execute(x)
                except FlowExceptionCR as e:
                    # this exception was already propagated,
                    # probably during the execution  of a node upstream in the flow
                    (exc_type, val) = _sys.exc_info()[:2]
                    prev = ''.join(_traceback.format_exception_only(e.__class__, e))
                    prev = prev[prev.find('\n') + 1:]
                    act = "\nWhile training node #%d (%s):\n" % (nodenr,
                                                                 str(self.flow[nodenr]))
                    err_str = ''.join(('\n', 40 * '=', act, prev, 40 * '='))
                    raise FlowException(err_str)
                except Exception as e:
                    # capture any other exception occured during training.
                    self._propagate_exception(e, nodenr)
        if empty_iterator:
            if self.flow[-1].get_current_train_phase() == 1:
                err_str = ("The training data iteration "
                           "could not be repeated for the "
                           "second training phase, you probably "
                           "provided an iterator instead of an "
                           "iterable.")
                raise FlowException(err_str)
            else:
                err_str = ("The training data iterator "
                           "is empty.")
                raise FlowException(err_str)
        self._stop_training_hook()
        if self.flow[-1].get_remaining_train_phase() > 1:
            # close the previous training phase
            self.flow[-1].stop_training()

    def _train_check_iterables(self, data_iterables):
        """Return the data iterable after some checks and sanitizing.

        Note that this method does not distinguish between iterables and
        iterators, so this must be taken care of later.
        """

        # if a single array is given, nodes are trained
        # incrementally if it is a 2D array or block incrementally if it
        # is a 3d array (num_blocks, block_size, dim).
        if isinstance(data_iterables, numx.ndarray):
            if data_iterables.ndim == 2:
                data_iterables = data_iterables[:, mdp.numx.newaxis, :]
            return data_iterables

        # check it it is an iterable
        if (data_iterables is not None) and (not hasattr(data_iterables, '__iter__')):
            err = "data_iterable is not an iterable."
            raise FlowException(err)

        return data_iterables

    def train(self, data_iterables):
        """Train all trainable nodes in the flow.

        'data_iterables' is a single iterable (including generator-type iterators if
        the last node has no multiple training phases) that must return data
        arrays to train nodes (so the data arrays are the 'x' for the nodes).
        Note that the data arrays are processed by the nodes
        which are in front of the node that gets trained,
        so the data dimension must match the input dimension of the first node.

        'data_iterables' can also be a 2D or a 3D numpy array. A 2D array trains
        all the nodes incrementally, while a 3D array supports online training
        in batches (=shape[1]).

        'data_iterables' can also return a list or a tuple, where the first entry is
        'x' and the rest are the required args for training all the nodes in
        the flow (e.g. for supervised training).

        (x, (node-0 args), (node-1 args), ..., (node-n args)) - args for n nodes

        if say node-i does not require any args, the provided (node-i args) are ignored.
        So, one can simply use None for the nodes that do not require args.

        (x, (node-0 args), ..., None, ..., (node-n args)) - No args for the ith node.

        """

        data_iterables = self._train_check_iterables(data_iterables)

        if self.verbose:
            strn = [str(self.flow[i]) for i in xrange(len(self.flow))]
            print("Training nodes %s simultaneously" % strn)
        self._train_nodes(data_iterables)

        # close training the terminal node only if it is a trainable Node
        if not isinstance(self.flow[-1], mdp.OnlineNode):
            self._close_last_node()

    # private container methods

    def _check_value_type_is_online_or_nontrainable_node(self, value):
        # valid: onlinenode, trained or non-trainable nodes
        # invalid: trainable but not an online node
        if not isinstance(value, mdp.Node):
            raise TypeError("flow item must be a Node instance and not %s" % type(value))
        elif isinstance(value, mdp.OnlineNode):
            # OnlineNode
            pass
        else:
            # Node
            if not value.is_training():
                # trained or non-trainable node
                pass
            else:
                # trainable but not an OnlineNode
                raise TypeError("flow item must either be an OnlineNode instance, a trained or a non-trainable Node.")

    def _check_compatibility(self, flow):
        [self._check_value_type_is_online_or_nontrainable_node(item) for item in flow[:-1]]
        # terminal node can be a trainable Node whose training hasn't finished.
        self._check_value_type_isnode(flow[-1])

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            [self._check_value_type_is_online_or_nontrainable_node(item) for item in value]
        else:
            self._check_value_type_is_online_or_nontrainable_node(value)

        flow_copy = list(self.flow)
        flow_copy[key] = value
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        self._check_compatibility(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy
        self._train_arg_keys_list, self._train_args_needed_list = self._get_required_train_args_from_flow(flow_copy)

    def __delitem__(self, key):
        flow_copy = list(self.flow)
        del flow_copy[key]
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        self._check_compatibility(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy
        self._train_arg_keys_list, self._train_args_needed_list = self._get_required_train_args_from_flow(flow_copy)

    def __add__(self, other):
        # append other to self
        if isinstance(other, mdp.Flow):
            flow_copy = list(self.flow).__add__(other.flow)
            # check OnlineFlow compatibility
            self._check_compatibility(flow_copy)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        elif isinstance(other, mdp.Node):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            # check OnlineFlow compatibility
            self._check_compatibility(flow_copy)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        else:
            err_str = ('can only concatenate OnlineFlow or flow with trained (or non-trainable) nodes'
                       ' (not \'%s\') to OnlineFlow' % type(other).__name__)
            raise TypeError(err_str)

    def __iadd__(self, other):
        # append other to self
        if isinstance(other, mdp.Flow):
            self.flow += other.flow
        elif isinstance(other, mdp.Node):
            self.flow.append(other)
        else:
            err_str = ('can only concatenate flow or node'
                       ' (not \'%s\') to flow' % type(other).__name__)
            raise TypeError(err_str)
        self._check_compatibility(self.flow)
        self._check_nodes_consistency(self.flow)
        self._train_arg_keys_list, self._train_args_needed_list = self._get_required_train_args_from_flow(self.flow)
        return self

    # public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]
        self._check_nodes_consistency(self.flow)
        self._check_compatibility(self.flow)
        self._train_arg_keys_list, self._train_args_needed_list = self._get_required_train_args_from_flow(self.flow)

    def extend(self, x):
        """flow.extend(iterable) -- extend flow by appending
        elements from the iterable"""
        if not isinstance(x, mdp.Flow):
            err_str = ('can only concatenate flow'
                       ' (not \'%s\') to flow' % type(x).__name__)
            raise TypeError(err_str)
        self[len(self):len(self)] = x
        self._check_nodes_consistency(self.flow)
        self._check_compatibility(self.flow)
        self._train_arg_keys_list, self._train_args_needed_list = self._get_required_train_args_from_flow(self.flow)

    def insert(self, i, x):
        """flow.insert(index, node) -- insert node before index"""
        self[i:i] = [x]
        self._check_nodes_consistency(self.flow)
        self._check_compatibility(self.flow)
        self._train_arg_keys_list, self._train_args_needed_list = self._get_required_train_args_from_flow(self.flow)


class CircularOnlineFlowException(mdp.MDPException):
    """Base class for exceptions in Flow subclasses."""
    pass


class CircularOnlineFlow(OnlineFlow):
    """A 'CircularOnlineFlow' is a cyclic sequence of online/non-trainable nodes that are trained and executed
    together to form a more complex algorithm. This type of flow is useful when one needs to train the nodes
    internally for several iterations using stored inputs before training with the next external input.

    input=x(i) -> [node1, node2, node3] -> output=y'     ----External input training----         (1 iteration)

    input=y'   -> [node1, node2, node3] -> output=y'     ---- Internal
    ...                                                               input                      (n-1 iterations)
    input=y'   -> [node1, node2, node3] -> output=y(i)                      training----

    This type of processing is especially useful for implementing control algorithms, reinforcement learning, etc.

    The input and the output nodes of the flow can be changed at any time using the functions "set_input_node"
    and "set_output_node".

    Examples:

    If input node and output node are set equal to node1 :

     Input -> [node1] -> output
             /       \
          [node3] <- [node2]

    If input node = node1 and output node is equal to node2

     Input -> [node1] -> [node2] -> output
                    \    /
                   [node3]

    CircularOnlineFlow also supports training nodes while ignoring external input data at all times. In this case,
    the flow iterates everytime using the previously stored output as the input. Input data can be arbitrary (with same
    input_dim and dtype) and is only used as a clock signal to trigger flow processing.
    See the docstring of the train method for more information about different training types.

    Crash recovery is optionally available: in case of failure the current
    state of the flow is saved for later inspection.

    CircularOnlineFlow objects are Python containers. Most of the builtin 'list'
    methods are available. CircularOnlineFlow can be saved or copied using the
    corresponding 'save' and 'copy' methods.

    """

    def __init__(self, flow, crash_recovery=False, verbose=False):
        """
        flow - a list of nodes.
        """
        super(CircularOnlineFlow, self).__init__(flow, crash_recovery, verbose)
        self.flow = _deque(flow)  # A circular queue of the flow

        # a variable to the set the number of internal flow iteration for each data point.
        self._flow_iterations = 1

        # set the last node of the list as the default output node.
        self.output_node_idx = len(self.flow) - 1

        # a variable to store inputs for internal train iterations
        self._stored_input = None

        # a flag when set ignores the input data (uses stored input instead).
        self._ignore_input = False

    def set_stored_input(self, x):
        """CircularOnlineFlow also supports training nodes while ignoring external input data at all times.
        In this case, the flow iterates every time using an initially stored input that can be set using this method.
        """
        if self.flow[0].input_dim is not None:
            if x.shape[-1] != self.flow[0].input_dim:
                raise CircularOnlineFlowException(
                    "Dimension mismatch! should be %d, given %d" % (self.flow[0].input_dim, x.shape[-1]))
            self._stored_input = x

    def get_stored_input(self):
        """return the current stored input"""
        return self._stored_input

    def ignore_input(self, flag):
        """ CircularOnlineFlow also supports training nodes while ignoring external input data at all times.
        This mode is enabled/disabled using this method. See train method docstring for information on
        different training modes.
        """
        self._ignore_input = flag

    def set_flow_iterations(self, n):
        """This method sets the number of total flow iterations:
        If self._ignore_input is False, then the total flow iterations = 1 external + (n-1) internal iterations.
        If self._ignore input is True, then the total flow iterations = n internal iterations.
        See train method docstring for information on different training modes.
        """
        self._flow_iterations = n

    def _train_nodes(self, data_iterables):
        for x in data_iterables:
            if self._ignore_input:
                # ignore external input
                x = self.get_stored_input()
            # train the loop for 'self.flow_iterations' iterations
            _iters = xrange(self._flow_iterations)
            if self.verbose:
                _iters = mdp.utils.progressinfo(_iters)
            for _ in _iters:
                for nodenr in xrange(len(self.flow)):
                    try:
                        node = self.flow[nodenr]
                        if node.is_training():
                            node.train(x)
                        x = node.execute(x)
                    except FlowExceptionCR as e:
                        # this exception was already propagated,
                        # probably during the execution  of a node upstream in the flow
                        (exc_type, val) = _sys.exc_info()[:2]
                        prev = ''.join(_traceback.format_exception_only(e.__class__, e))
                        prev = prev[prev.find('\n') + 1:]
                        act = "\nWhile training node #%d (%s):\n" % (nodenr,
                                                                     str(self.flow[nodenr]))
                        err_str = ''.join(('\n', 40 * '=', act, prev, 40 * '='))
                        raise FlowException(err_str)
                    except Exception as e:
                        # capture any other exception occured during training.
                        self._propagate_exception(e, nodenr)
                self._stored_input = x

    def train(self, data_iterables):
        """Train all trainable-nodes in the flow.

        'data_iterables' is a single iterable (including generator-type iterators)
        that must return data arrays to train nodes (so the data arrays are the 'x'
        for the nodes). Note that the data arrays are processed by the nodes
        which are in front of the node that gets trained,
        so the data dimension must match the input dimension of the first node.

        'data_iterables' can also be a 2D or a 3D numpy array. A 2D array trains
        all the nodes incrementally, while a 3D array supports online training
        in batches (=shape[1]).

        Circular flow does not support passing training arguments.

        There are three ways that the training can proceed based on the values
        of self._ignore_input (default=False, can be set via 'ignore_input' method) and
        self._flow_iterations argument (default=1, can be set via 'set_flow_iterations' method).

        1) self._ignore_input = False, self._flow_iterations = 1 (default case)

            This is functionally similar to the standard OnlineFlow.
            Each data array returned by the data_iterables is used to
            train the nodes simultaneously.

        2) self._ignore_input = False, self._flow_iterations > 1
            For each data_array returned by the data_iterables, the flow
            trains 1 loop with the data_array and 'self._flow_iterations-1'
            loops with the updating stored inputs.

        3) self._ignore_input = True, self._flow_iterations > 1
            The data_arrays returned by the data_iterables are ignored, however,
            for each data_array, the flow trains 'self._flow_iterations' loops
            with the updating stored inputs.

        """

        if self.verbose:
            strn = [str(self.flow[i]) for i in xrange(len(self.flow))]
            if self._ignore_input:
                print("Training nodes %s internally using the stored inputs "
                      "for %d loops" % (strn, self._flow_iterations))
            else:
                print("Training nodes %s using the given inputs and %d loops "
                      "internally for each data point" % (strn, self._flow_iterations))

        data_iterables = self._train_check_iterables(data_iterables)
        self._train_nodes(data_iterables)

    def execute(self, iterable, nodenr=None):
        """Process the data through all nodes between input and the output node.
        This is functionally similar to the execute method of an OnlineFlow.

        'iterable' is an iterable or iterator (note that a list is also an
        iterable), which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.
        """
        if nodenr is None:
            nodenr = self.output_node_idx
        return super(CircularOnlineFlow, self).execute(iterable, nodenr)

    def _inverse_seq(self, x):
        # Successively invert input data 'x' through all nodes backwards from the output node to the input node.
        flow = self.flow[:self.output_node_idx]
        for i in range(len(flow) - 1, -1, -1):
            try:
                x = flow[i].inverse(x)
            except Exception as e:
                self._propagate_exception(e, i)
        return x

    def set_input_node(self, node_idx):
        """Set the input node of the flow"""
        if (node_idx > len(self.flow)) or (node_idx < 0):
            raise CircularOnlineFlowException(
                "Accepted 'node_idx' values: 0 <= node_idx < %d, given %d" % (len(self.flow), node_idx))
        self.flow.rotate(-node_idx)
        self.output_node_idx = (self.output_node_idx - node_idx) % len(self.flow)

    def set_output_node(self, node_idx):
        """Set the output node of the flow"""
        if (node_idx > len(self.flow)) or (node_idx < 0):
            raise CircularOnlineFlowException(
                "Accepted 'node_idx' values: 0 <= node_idx < %d, given %d" % (len(self.flow), node_idx))
        self.output_node_idx = node_idx

    def _check_compatibility(self, flow):
        [self._check_value_type_is_online_or_nontrainable_node(item) for item in flow]

    def reset_output_node(self):
        """Resets the output node to the last node of the provided node sequence"""
        self.output_node_idx = len(self.flow) - 1

    # private container methods

    def __setitem__(self, key, value):
        super(CircularOnlineFlow, self).__setitem__(key, value)
        self.flow = _deque(self.flow)
        if (key.start < self.output_node_idx) and (self.output_node_idx < key.stop()):
            print('Output node is replaced! Resetting the output node.')
            self.reset_output_node()

    def __getitem__(self, key):
        if isinstance(key, slice):
            flow_copy = list(self.flow)
            flow_slice = flow_copy[key]
            self._check_nodes_consistency(flow_slice)
            return self.__class__(flow_slice)
        else:
            return self.flow[key]

    def __delitem__(self, key):
        super(CircularOnlineFlow, self).__delitem__(key)
        self.flow = _deque(self.flow)
        if (key.start < self.output_node_idx) and (self.output_node_idx < key.stop()):
            print('Output node deleted! Resetting the output node to the default last node.')
            self.reset_output_node()
        elif self.output_node_idx > key.stop():
            self.set_output_node(self.output_node_idx - key.stop + key.start)

    def __add__(self, other):
        # append other to self
        if isinstance(other, OnlineFlow):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            self._check_compatibility(flow_copy)
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        elif isinstance(other, mdp.OnlineNode):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            # check onlineflow compatibility
            self._check_compatibility(flow_copy)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        else:
            err_str = ('can only concatenate OnlineFlow or OnlineNode'
                       ' (not \'%s\') to CircularOnlineFlow' % type(other).__name__)
            raise TypeError(err_str)

    def __iadd__(self, other):
        # append other to self
        if isinstance(other, OnlineFlow):
            self.flow += other.flow
        elif isinstance(other, mdp.OnlineNode):
            self.flow.append(other)
        else:
            err_str = ('can only concatenate OnlineFlow or OnlineNode'
                       ' (not \'%s\') to CircularOnlineFlow' % type(other).__name__)
            raise TypeError(err_str)
        self._check_compatibility(self.flow)
        self._check_nodes_consistency(self.flow)
        return self

    # public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]
        self._check_nodes_consistency(self.flow)
        self._check_compatibility(self.flow)

    def extend(self, x):
        """flow.extend(iterable) -- extend flow by appending
        elements from the iterable"""
        if not isinstance(x, mdp.Flow):
            err_str = ('can only concatenate OnlineFlow'
                       ' (not \'%s\') to CircularOnlineFlow' % type(x).__name__)
            raise TypeError(err_str)
        self[len(self):len(self)] = x
        self._check_nodes_consistency(self.flow)
        self._check_compatibility(self.flow)

    def insert(self, i, x):
        """flow.insert(index, node) -- insert node before index"""
        self[i:i] = [x]
        self._check_nodes_consistency(self.flow)
        self._check_compatibility(self.flow)

        if self.output_node_idx >= i:
            self.set_output_node(self.output_node_idx + 1)
