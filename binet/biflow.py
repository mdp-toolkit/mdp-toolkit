"""
Flow handling classes for a BiNet.

The central class is a BiFlow, which implements all the flow handling options
offered by the BiNode class (see binode.py for a description).
"""

import itertools

import mdp
n = mdp.numx

from binode import BiNode

# this target value tells the flow to abort and return the current values 
EXIT_TARGET = "exit"


class NoneIterable(object):
    """Iterable for an infinite sequence of Nones."""
    
    def __iter__(self):
        while True:
            yield None


class BiFlowException(mdp.FlowException):
    """Exception for BiFlow problems."""
    pass


class MessageResultContainer(object):
    """Store and combine msg output chunks from a BiNode.
    
    It is for example used when the flow execution yields msg output, which has
    to be joined for the end result.
    """
    
    def __init__(self):
        """Initialize the internal storage variables."""
        self._msg_results = dict()  # all none array message results
        self._msg_array_results = dict()  # result dict for arrays
        
    def add_message(self, msg):
        """Add a single msg result to the combined results.
        
        msg must be either a dict of results or None. numpy arrays will be
        transformed to a single numpy array in the end. For all other types the
        addition operator will be used to combine results (i.e., lists will be
        appended, single integers will be summed over).
        """
        if msg:
            for key in msg:
                if type(msg[key]) is n.ndarray:
                    if key not in self._msg_array_results:
                        self._msg_array_results[key] = []
                    self._msg_array_results[key].append(msg[key])
                else:
                    if key not in self._msg_results:
                        self._msg_results[key] = msg[key]
                    else:
                        try:
                            self._msg_results[key] += msg[key]
                        except:
                            err = ("Could not combine final msg results "
                                   "in BiFlow.")
                            raise BiFlowException(err)
    
    def get_message(self):
        """Return the msg which combines all the msg results."""
        # move array results from _msg_array_results to _msg_results
        for key in self._msg_array_results:
            if key in self._msg_results:
                err = ("A key in the msg results is used with " 
                       "different data types.")
                raise BiFlowException(err)
            else:
                self._msg_results[key] = n.concatenate(
                                                self._msg_array_results[key])
        if self._msg_results:
            return self._msg_results
        else:
            return None
        

# TODO: make sure that isinstance(str, target) is never used, so that in
#    principle any object could be used (the might have to overwrite the
#    _node_with_id method?

class BiFlow(mdp.Flow):
    """BiNet version of a flow, which supports jumps between nodes.
    
    This capabilities can be used by classes derived from BiNode. 
    
    Normal nodes can also be used in this flow, the msg argument is skipped 
    for these. Normal nodes can be also jump targets, but only when a relative
    target index is used (since they do not support node ids).
    """
    
    def __init__(self, flow, verbose=False, **kwargs):
        kwargs["crash_recovery"] = False
        super(BiFlow, self).__init__(flow=flow, verbose=verbose, **kwargs)
    
    ### Basic Methods from Flow. ###
    
    def train(self, data_iterables, msg_iterables=None, 
              stop_messages=None):
        """Train the nodes in the flow.
        
        The nodes will be trained according to their place in the flow.
        
        data_iterables -- Sequence of iterables with the training data for each
            trainable node. Can also be a single array or None.
        msg_iterables -- Sequence of iterables with the msg training data 
            for each trainable node.
        stop_messages -- Sequence of messages for stop_training.
        
        Note that the type and iterator length of the data iterables is taken 
        as reference, so the message iterables are assumed to have the 
        same length.
        """
        # Note: When this method is updated BiCheckpointFlow should be updated
        #    as well.
        self._bi_reset()  # normaly not required, just for safety
        data_iterables, msg_iterables = self._sanitize_training_iterables(
                                            data_iterables=data_iterables, 
                                            msg_iterables=msg_iterables)
        if stop_messages is None:
            stop_messages = [None] * len(data_iterables)
        # train each Node successively
        for i_node in range(len(self.flow)):
            if self.verbose: 
                print ("training node #%d (%s)" % 
                       (i_node, str(self.flow[i_node])))
            self._train_node(data_iterables[i_node], i_node, 
                             msg_iterables[i_node], stop_messages[i_node])
            if self.verbose: 
                print "training finished"
                
    def _train_node(self, iterable, nodenr, msg_iterable=None, 
                    stop_msg=None):
        """Train a particular node.
        
        nodenr -- index of the node to be trained
        msg_iterable -- optional msg data for the training
            Note that the msg is only passed to the Node if it is an instance
            of BiNode.
        stop_msg -- optional msg data for stop_training
            Note that the message is only passed to the Node if the msg is not
            None, so for a normal node the msg has to be None.
        
        Note: unlike the normal mdp.Flow we do no exception handling here.
        """
        if not self.flow[nodenr].is_trainable():
            return
        iterable, msg_iterable, _ = self._sanitize_iterables(iterable,
                                                             msg_iterable)
        while True:
            if not self.flow[nodenr].get_remaining_train_phase():
                break
            self._train_node_single_phase(iterable, nodenr, 
                                          msg_iterable, stop_msg)
            
    def _train_node_single_phase(self, iterable, nodenr, 
                                 msg_iterable, stop_msg=None):
        """Perform a single training phase for a given node.
        
        This method should be only called internally in BiFlow.
        """
        i_node = nodenr
        empty_iterator = True
        for (x, msg) in itertools.izip(iterable, msg_iterable):
            empty_iterator = False
            ## execute the flow until the nodes return value is right
            target = 0
            while True:
                result = self._execute_seq(x, msg, target=target,
                                           stop_at_node=i_node)
                ## check the execution result, target should be True
                if (not isinstance(result, tuple)) or (len(result) != 3):
                    err = ("The Node to be trained was not reached " +
                           "during training, last result: " + str(result))
                    raise BiFlowException(err)
                elif result[2] is True:
                    x = result[0]
                    msg = result[1]
                else:
                    err = ("Target node not found in flow during " +
                           "training, last target value: " + str(result[2]))
                    raise BiFlowException(err)
                ## perform node training
                if isinstance(self.flow[i_node], BiNode):
                    result = self.flow[i_node].train(x, msg)
                    if not result:
                        break  # training is done for this chunk
                else:
                    self.flow[i_node].train(x)
                    break
                ## training execution continues, interpret result
                if not isinstance(result, tuple):
                    x = result
                    msg = None
                    target = None
                elif len(result) == 2:
                    x, msg = result
                    target = 1
                elif len(result) == 3:
                    x, msg, target = result
                else:        
                    err = ("Node produced invalid return value " +
                           "during training: " + str(result))
                    raise BiFlowException(err)
                target = self._target_to_index(target, i_node)
            self._bi_reset()
        if empty_iterator:
            if self.flow[i_node].get_current_train_phase() == 1:
                err_str = ("The training data iteration for node "
                           "no. %d could not be repeated for the "
                           "second training phase, you probably "
                           "provided an iterable instead of an "
                           "iterable." % (i_node+1))
                raise BiFlowException(err_str)
            else:
                err = ("The training data iterable for node "
                       "no. %d is empty." % (i_node+1))
                raise BiFlowException(err)
        # unlike the normal mdp.Flow we always close the training
        # otherwise stop_message propagation would be difficult
        self._stop_training_hook()
        if stop_msg is None:
            result = self.flow[i_node].stop_training()
        else:
            result = self.flow[i_node].stop_training(stop_msg)
        if result is not None:
            if len(result) == 2:
                result = self._stop_message_seq(msg=result[0],
                                                target=result[1],
                                                current_node=i_node)
                if isinstance(result, tuple):
                    err = ("Stop message target node not found in flow " +
                           "during training, last result: " + 
                           str(result))
                    raise BiFlowException(err)
            else:
                err = ("Node produced invalid return value for " +
                       "stop_training: " + str(result))
                raise BiFlowException(err)
        self._bi_reset()
        
    def execute(self, iterable, msg_iterable=None, target_iterable=None):
        """Execute the flow and return y or (y, msg).
        
        iterable -- Can be an iterable or iterator for arrays, a single array 
            or None. In the last two cases it is assumed that msg is a single
            message as well.
        msg_iterable -- Can be an iterable or iterator or a single message
            (but only if iterable is a single array or None).
        target_iterable -- Like msg_iterable, but for target.
        
        Note that the type and iteration length of iterable is taken as 
        reference, so msg is assumed to have the same length.
        
        If msg results are found and if iteration is used then the BiFlow
        tries to join the msg results (and concatenate in the case of arrays).
        """
        self._bi_reset()  # normaly not required, just for safety
        iterable, msg_iterable, target_iterable = \
            self._sanitize_iterables(iterable, msg_iterable, target_iterable)
        y_results = None
        msg_results = MessageResultContainer()
        empty_iterator = True
        for (x, msg, target) in itertools.izip(iterable, msg_iterable,
                                               target_iterable):
            empty_iterator = False
            ## execute the flow until the nodes return value is right
            if target is None:
                target = 0
            # loop to deal with eventual intermediate global messages
            while True:
                result = self._execute_seq(x=x, msg=msg, target=target)
                if not isinstance(result, tuple):
                    y = result
                    msg = None
                    break
                elif (len(result) == 2):
                    y, msg = result
                    self._global_message_seq(msg)
                    break
                elif (len(result) == 3) and (result[2] == EXIT_TARGET):
                    y, msg = result[:2]
                    self._global_message_seq(msg)
                    break
                elif len(result) == 3:
                    err = ("Target node not found in flow during execute," + 
                           " last result: " + str(result))
                    raise BiFlowException(err)
                else:
                    err = ("BiNode execution returned invalid result type: " + 
                           result)
                    raise BiFlowException(err)    
            self._bi_reset()
            if msg:
                msg_results.add_message(msg)
            # check if all y have the same type and store it
            # note that the checks for msg are less restrictive
            if y is not None:
                if y_results is None:
                    y_results = [y]
                elif y_results is False:
                    err = "Some but not all y return values were None."
                    raise BiFlowException(err)
                else:
                    y_results.append(y)
            else:
                if y_results is None:
                    y_results = False
                else:
                    err = "Some but not all y return values were None."
                    raise BiFlowException(err)
        if empty_iterator:
            err = ("The execute data iterable is empty.")
            raise BiFlowException(err)
        # consolidate results
        if y_results:
            y_results = n.concatenate(y_results)
        result_msg = msg_results.get_message()
        if result_msg:
            return (y_results, result_msg)
        else:
            return y_results
    
    def __call__(self, iterable, msg_iterable=None):
        """Calling an instance is equivalent to call its 'execute' method."""
        return self.execute(iterable, msg_iterable=msg_iterable)
        
    ### New Methods for BiNet. ###
    
    def _bi_reset(self):
        """Reset the nodes and internal flow variables."""
        self._global_message_emitter = None
        for node in self.flow:
            if isinstance(node, BiNode):
                node.bi_reset()
    
    def _request_node_id(self, node_id):
        """Return first hit of _request_node_id on internal nodes.
        
        So _request_node_id is called for all nodes in the flow until a return
        value is not None. If no such node is found the return value is None.
        """
        for node in self.flow:
            if isinstance(node, BiNode):
                found_node = node._request_node_id(node_id)
                if found_node:
                    return found_node
        return None

    ## container special methods to support node_id
    
    def __getitem__(self, key):
        if isinstance(key, str):
            item = self._request_node_id(key)
            if item is None:
                err = ("This biflow contains no node with with the id " +
                       str(key))
                raise KeyError(err)
            return item
        else:
            return super(BiFlow, self).__getitem__(key)
        
    def __setitem__(self, key, value):
        if isinstance(key, str):
            err = "Setting nodes by node_id is not supported."
            raise BiFlowException(err)
        else:
            super(BiFlow, self).__setitem__(key, value)
    
    def __delitem__(self, key):
        if isinstance(key, str):
            err = "Deleting nodes by node_id is not supported."
            raise BiFlowException(err)
        else:
            super(BiFlow, self).__delitem__(key)
    
    def __contains__(self, key):
        if isinstance(key, str):
            if self._request_node_id(key) is not None:
                return True
            else:
                return False
        else:
            return super(BiFlow, self).__contains__(key)
    
    ### Flow Implementation Methods ###
    
    def _sanitize_training_iterables(self, data_iterables, msg_iterables):
        """Check and adjust the training iterable list."""
        if data_iterables is None:
            if msg_iterables is None:
                err = ("Both the training data and the training messages are "
                       "None.")
                raise BiFlowException(err)
            else:
                data_iterables = [None] * len(self.flow)
        elif isinstance(data_iterables, n.ndarray):
            data_iterables = [[data_iterables]] * len(self.flow)
            # the form of msg_iterables follows that of data_iterables
            msg_iterables = [[msg_iterables]] * len(data_iterables)
        else:   
            data_iterables = self._train_check_iterables(data_iterables)
            if msg_iterables is None:
                msg_iterables = [None] * len(self.flow)
            else:
                msg_iterables = self._train_check_iterables(msg_iterables)
        return data_iterables, msg_iterables
    
    def _sanitize_iterables(self, iterable, msg_iterable, target_iterable=None):
        """Check and adjust a data, message and target iterable."""
        # TODO: maybe add additional checks
        if isinstance(iterable, n.ndarray):
            iterable = [iterable]
            msg_iterable = [msg_iterable]
            target_iterable = [target_iterable]
        elif iterable is None:
            if msg_iterable is None:
                err = "Both the data and the message iterable is None."
                raise BiFlowException(err)
            else:
                iterable = NoneIterable()
                if isinstance(msg_iterable, dict):
                    msg_iterable = [msg_iterable]
                    target_iterable = [target_iterable]
        else:
            if msg_iterable is None:
                msg_iterable = NoneIterable()
            if target_iterable is None:
                target_iterable = NoneIterable()
        return iterable, msg_iterable, target_iterable
    
    def _target_to_index(self, target, current_node=0):
        """Return the target node index corresponding to the target code.
        
        If the string id target node is not found in this flow then the string
        is returned without alteration.
        
        When a relative index is given it is translated to an absolute index 
        and it is checked if it is in the allowed range.
        
        target -- Can be a string node id, a relative index or None (which
            is interpreted as 1).
        current_node -- If target is specified as a relative index then this
            node index is used to translate the target to the absolute node 
            index (otherwise it has no effect).
        """
        if target == EXIT_TARGET:
            return EXIT_TARGET
        if target is None:
            target = 1
        if not isinstance(target, int):
            for i_node, node in enumerate(self.flow):
                if isinstance(node, BiNode) and node._request_node_id(target):
                    return i_node
            # no matching node was found
            return target
        else:
            absolute_index = current_node + target
            if absolute_index < 0:
                err = "Target int value references node at position < 0."
                raise BiFlowException(err)
            elif absolute_index >= len(self.flow):
                err = ("Target int value references a node"
                       " beyond the flow length (target " + str(target) +
                       ", current node " + str(current_node) + ").")
                raise BiFlowException(err)
            return absolute_index
    
    def _execute_seq(self, x, msg=None, target=0, stop_at_node=None):
        """Execute the whole flow as far as possible.
        
        target -- Can specify a node_id where the excecution is supposed to
            start or can be node_id (_target_to_index is used to convert
            target).
        stop_at_node -- Node index where the execution should stop. The input
            values for this node are returned in this case in the form
            (x, msg, target) with target being set to True.
        
        If the end of the flow is reached then the return value is y
        or (y, msg).
        If the an execution target node is not found then (x, msg, target) is
        returned. 
        If an untrained node or a node in training is encountered, then a 
        exception is raised.
        
        If a normal Node (not derived from BiNode) is encountered then the
        current msg is simply carried forward around it. 
        """
        ## this method is also used by other classes, like BiFlowNode
        i_node = self._target_to_index(target)
        if not isinstance(i_node, int):
            # target not found in this flow
            return x, msg, target
        while i_node != stop_at_node:
            if isinstance(self.flow[i_node], BiNode):
                result = self.flow[i_node].execute(x, msg)
                # check the type of the result
                if type(result) is not tuple:
                    x = result
                    msg = None
                    target = 1
                elif len(result) == 2:
                    x, msg = result
                    target = 1
                elif len(result) == 3:
                    x, msg, target = result
                else:
                    err = ("BiNode execution returned invalid result type: " + 
                           result)
                    raise BiFlowException(err)
            else:
                # just a normal MDP node
                x = self.flow[i_node].execute(x)
                # note that the message is carried forward unchanged
                target = 1
            # check if we should leave the flow
            if (target == 1) and (i_node + 1 == len(self.flow)):
                if not msg:
                    return x
                else:
                    return (x, msg)
            elif (target == -1) and (i_node == 0):
                return x, msg, target
            # update i_node to the target node
            i_node = self._target_to_index(target, i_node)
            if not isinstance(i_node, int):
                # target not found in this flow
                # this is also the exit point when EXIT_TARGET is given
                return x, msg, target
        # reached stop_at_node, signal this by returning target value True
        return x, msg, True
            
    def _stop_message_seq(self, msg, target, current_node=0):
        """Propagate a stop_message through the flow.
        
        If the bi_stop_message terminated there is no return value. If a target
        node was not found along the way the return value is (msg, target).
        """
        ## note that the code is almost identical to _message_seq
        i_node = self._target_to_index(target, current_node)
        if not isinstance(i_node, int):
            # target not found in this flow
            return msg, target
        while True:
            if i_node is None:
                # target not found in this flow
                return msg, target
            if not isinstance(self.flow[i_node], BiNode):
                err = ("Called stop_message on a normal node, target was: " +
                       str(self.flow[i_node]))
                raise BiFlowException(err)
            result = self.flow[i_node].stop_message(msg)
            # check the type of the result
            if result is None:
                # reached end of message sequence
                return None
            elif len(result) != 2:
                err = ("BiNode stop_message returned tuple " +
                       "of length %d instead of 2." % len(result))
                raise BiFlowException(err)
            msg, target = result
            # update i_node to the target node
            i_node = self._target_to_index(target, i_node)
            if not isinstance(i_node, int):
                # target not found in this flow
                return msg, target
            
      
### Some useful flow classes. ###

class BiCheckpointFlow(BiFlow, mdp.CheckpointFlow):
    """Similar to normal checkpoint flow.
    
    The main difference is that even the last training phase of a 
    node is already closed before the checkpoint function is called (otherwise 
    it would be difficult to propagate bi_stop_message).
    """
    
    def train(self, data_iterables, checkpoints, msg_iterables=None,
              stop_messages=None):
        """Train the nodes in the flow.
        
        The nodes will be trained according to their place in the flow.
        
        Additionally calls the checkpoint function 'checkpoint[i]'
        when the training phase of node #i is over.
        A checkpoint function takes as its only argument the trained node.
        If the checkpoint function returns a dictionary, its content is
        added to the instance's dictionary.
        The class CheckpointFunction can be used to define user-supplied
        checkpoint functions.
        """
        self._bi_reset()  # normaly not required, just for safety
        data_iterables, msg_iterables = self._sanitize_training_iterables(
                                                 data_iterables=data_iterables, 
                                                 msg_iterables=msg_iterables)
        if stop_messages is None:
            stop_messages = [None] * len(data_iterables)
        checkpoints = self._train_check_checkpoints(checkpoints)
        # train each Node successively
        for i_node in range(len(self.flow)):
            if self.verbose: 
                print ("training node #%d (%s)" % 
                       (i_node, str(self.flow[i_node])))
            self._train_node(data_iterables[i_node], i_node, 
                             msg_iterables[i_node], stop_messages[i_node])
            if i_node <= len(checkpoints) and checkpoints[i_node] is not None:
                checkpoint_dict = checkpoints[i_node](self.flow[i_node])
                if dict: 
                    self.__dict__.update(checkpoint_dict)
            if self.verbose: 
                print "training finished"

