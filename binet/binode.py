"""
Special BiNode class derived from Node to allow complicated flow patterns.

Messages:
=========

The message argument 'msg' is either a dict or None (which is treated equivalent 
to an empty dict and is the default value). The message is automatically 
parsed against the method signature in the following way:

    normal key string -- Is copied if in signature and passed as kwarg.
    
    node_id=>key -- Is extracted (i.e. removed in original message) and passed 
        as a kwarg.
        
    If a node has a kwarg named msg then the whole remaining message is
    passed after parsing. It is then completely replaced by the result
    message (so one has to forward everything that should be kept).
        
The msg returned from the inner part of the method (e.g. _execute) is then used
to update the original message (so values can be overwritten).

Additionally there can be global message parts, which are used during the
global message phase:
    
    @key -- Is copied if key is in signature and passed as kwarg.
        
    node_id@key -- Is extracted (i.e. removed in original message) and passed 
        as a kwarg.
        
    If _global_message of has a kwarg named msg, then the whole remaining
    message is passed (and can be modified _in place_ by the node).
        
The global message phase phase is entered whenever there is no target
specified for the message in a branch or for the stop message. When training
terminates in a node, then any remaining global messages are also processed.
When the flow is exited during execution any remaining global messages are
processed as well.
        
If the message results in passing kwargs that are not args of the Node or if
args without default value are missing in the message, this will result in the
standard Python missing-arguments-exception (this is not checked by BiNode
itself). 


BiNode Return Value Options:
============================

 result for execute:
     (x, msg, target, branch_msg, branch_target)
     or shorter: x, (x, msg), (x, msg, target), (x, msg, target, branch_msg)
     
 result for train:
     terminate training for return types:
         None, msg, (msg, target)
     or continue execution:
         (x, msg, target, branch_msg, branch_target),
         (x, msg, target, branch_msg)
     
 result for stop_training
     (msg, target)
     or shorter form: msg
     
 result for message and stop_message:
     (msg, target)
     or shorter form msg
     
 result for global_message: 
     None
     

Magic keyword arguments:
========================

When the incoming message is parsed by the BiNode base class, some argument
keywords are treated in a special way:

 'msg' -- If any method like _execute accept a 'msg' keyword than the complete
     remaining message (after parsing the other keywords) is supplied. The
     message in the return value then completely replaces the original message
     (instead of only updating it). This way a node can completely control the
     message.
     
 'target' -- If any template method like execute finds a 'target' keyword in
     the message then this is used as the target value in the return value.
     However, if _execute then also returns a target value this overwrites the
     target value. In global_message calls 'target' has no special meaning and
     can be used like any other keyword.
     
"""

# TODO: store inspect.getargspec(self._stop_message)[0] in self,
#    that should be more efficient

# TODO: use of namedtuple for the return value?
#    see http://docs.python.org/library/collections.html#collections.namedtuple

# TODO: Implement internal checks for node output result?
#    Check that last element is not None? Use assume?
#    Should this match the level of output-checking of MDP?

import inspect

import mdp

# separator / flag strings for message keys
NODE_ID_KEY = "=>"  
GLOBAL_CHAR = "@"  # single char prefix for global keys
GLOBAL_ID_KEY = "@"
TARGET_KEY = "target"  # target keyword


class BiNodeException(mdp.NodeException):
    """Exception for BiNode problems."""
    pass


# methods that can overwrite docs:
DOC_METHODS = ['_train', '_stop_training', '_execute', '_inverse',
               '_message', '_stop_message', '_global_message']

class BiNodeMetaClass(mdp.NodeMetaclass):
    """Custom BiNode metaclass to deal with the special signature handling."""
    
    def __new__(mcl, classname, bases, members):
        for privname in DOC_METHODS:
            if privname in members:
                priv_info = mcl._get_infodict(members[privname])
                if not priv_info['doc']:
                    continue
                pubname = privname[1:]
                if pubname in members:
                    continue
                for base in bases:
                    ancestor = base.__dict__
                    if pubname in ancestor:
                        # preserve the signature (in addition to name)
                        pub_info = mcl._get_infodict(ancestor[pubname])
                        priv_info['name'] = pub_info['name']
                        priv_info['signature'] = pub_info['signature']
                        priv_info['argnames'] = pub_info['argnames']
                        priv_info['defaults'] = pub_info['defaults']
                        members[pubname] = mcl._wrap_func(ancestor[pubname], 
                                                          priv_info)
                        break
        return type.__new__(mcl, classname, bases, members)
    
    
class BiNode(mdp.Node):
    """Abstract base class for all bidirectional nodes.
    
    This class is not non-functional, since the arguments of the inherited
    _execute, _train and _stop_training methods are incompatible with the calls.
    So these methods (or _get_train_sequence instead of the last two) have to
    be overridden.
    
    Note hat this class can also be used as an Adapter / Mixin for normal nodes.
    This can for example be useful for nodes which require additional data
    arguments during training or execution. These can then be encapsulated in a
    messsage. Note that BiNode has to come first in the MRO to make all this
    work.
    """
    
    __metaclass__ = BiNodeMetaClass
    
    def __init__(self, node_id=None, stop_msg=None, **kwargs):
        """Initialize BiNode.
        
        _node_id -- None or string which identifies the node.
        stop_msg -- A msg dict which is send by stop_training.
            If the node has multiple training phases then this
            can also be a list with one entry for each training phase.
            This argument is added purely for convenience and should only be
            used for global messages.
        
        kwargs are forwarded via super to the next __init__ method
        in the MRO.
        """
        self._node_id = node_id
        self._stop_msg = stop_msg
        # this is used for the msg parsing by _matches_classname
        self._mro_classnames = [class_obj.__name__ 
                                for class_obj in self.__class__.mro()]
        super(BiNode, self).__init__(**kwargs)
        
    ### Modified template methods from mdp.Node. ###
    
    def execute(self, x, msg=None):
        """Return single value y or a result tuple.
        
        The possible result tuples are (y, msg), (y, msg, target),
        (y, msg, target, bi_msg) or (y, msg, target, branch_msg, branch_target).
        
        The outgoing msg carries forward the incoming message content, while 
        branch_msg starts blank.
        
        The last entry in a result tuple must not be None. 
        x can be None, then the usual checks are omitted.
        y can be None if the result is a tuple.
        
        This template method calls the corresponding _execute method.
        """
        if x is not None:
            self._pre_execution_checks(x)
            x = self._refcast(x)
        # analyze msg
        if msg is None:
            return self._execute(x)
        arg_keys = inspect.getargspec(self._execute)[0]
        msg, msg_kwargs, target = self._parse_message(msg, arg_keys)
        result = self._execute(x, **msg_kwargs)
        # overwrite result values if necessary and return
        if isinstance(result, tuple):
            if msg and result[1]:
                # combine outgoing msg and remaining msg values
                msg.update(result[1])
                result = (result[0], msg) + result[2:]
            if target is not None:
                if len(result) == 2:
                    result += (target,)
                elif result[2] is None:
                    result = result[:2] + (target,) + result[3:]
            return result
        else:
            # result is only single array
            if (not msg) and (target is None):
                return result
            elif target is None:
                return result, msg
            else:
                return result, msg, target
            
    def is_trainable(self):
        """Return the return value from super."""
        return super(BiNode, self).is_trainable()
    
    def train(self, x, msg=None):
        """Train and return None, msg or a result tuple.
        
        The training execution ends here for return values of:
            None, msg, (msg, target)
        The training execution continues for:
            (x, msg, target, branch_msg, branch_target),
            (x, msg, target, branch_msg),
            (x, msg, target)
        
        Any leftover msg is merged into the first result msg.
        
        This template method calls a _train(x, msg) method from self._train_seq.
        """
        # perform checks, adapted from Node.train
        if not self.is_trainable():
            raise mdp.IsNotTrainableException("This node is not trainable.")
        if not self.is_training():
            err = "The training phase has already finished."
            raise mdp.TrainingFinishedException(err)
        if x is not None:
            self._check_input(x)
        self._check_train_args(x, msg)        
        self._train_phase_started = True
        if x is not None:
            x = self._refcast(x)
        train_method = self._train_seq[self._train_phase][0]
        if msg is None:
            return train_method(x)
        arg_keys = inspect.getargspec(train_method)[0]
        msg, msg_kwargs, target = self._parse_message(msg, arg_keys)
        result = train_method(x, **msg_kwargs)
        # overwrite result values if necessary and return
        if isinstance(result, tuple) and (len(result) >= 3):
            # continue with training execution
            if msg and result[1]:
                msg.update(result[1])
                result = (result[0], msg) + result[2:]
            if target is not None:
                if len(result) == 2:
                    result += (target,)
                elif result[2] is None:
                    result = result[:2] + (target,) + result[3:]
        else:
            result = self._combine_message_result(result, msg, target)
        return result
    
    def stop_training(self, msg=None):
        """Stop training phase and return None, msg or (msg, target).
        
        The result tuple is then used to call stop_message on the target node.
        The outgoing msg carries forward the incoming message content.
        
        This template method calls a _stop_training method from self._train_seq.
        
        If a stop_msg was given in __init__ then it is incorporated into the
        outgoing msg (but can be overwritten by the _stop_training msg output).
        """
        if self.is_training() and self._train_phase_started == False:
            raise mdp.TrainingException("The node has not been trained.")
        if not self.is_training():
            err = "The training phase has already finished."
            raise mdp.TrainingFinishedException(err)
        # get message information for stored stop message
        if isinstance(self._stop_msg, tuple):
            stored_stop_msg = self._stop_msg[self._train_phase]
        else:
            stored_stop_msg = self._stop_msg
        stop_method = self._train_seq[self._train_phase][1]
        if not msg:
            result = stop_method()
            msg = stored_stop_msg
            target = None
        else:
            arg_keys = inspect.getargspec(stop_method)[0]
            msg, msg_kwargs, target = self._parse_message(msg, arg_keys)
            result = stop_method(**msg_kwargs)
            if stored_stop_msg:
                msg.update(stored_stop_msg)
        # overwrite result values if necessary
        result = self._combine_message_result(result, msg, target)
        # close the current phase
        self._train_phase += 1
        self._train_phase_started = False
        # check if we have some training phase left
        if self.get_remaining_train_phase() == 0:
            self._training = False
        return result
    
    ### New methods for node messaging. ###
    
    def message(self, msg=None):
        """Receive message and return None, msg or (msg, target).
        
        If the return value is (msg, target) then message(msg) is called on
        the target node.
        
        This template method calls the _message method.
        """
        if not msg:
            return self._message()
        arg_keys = inspect.getargspec(self._message)[0]
        msg, msg_kwargs, target = self._parse_message(msg, arg_keys)
        result = self._message(**msg_kwargs)
        return self._combine_message_result(result, msg, target)

    def _message(self):
        """Hook method, overwrite when needed. 
        
        This default implementation only raises an exception.
        """
        err = "This node does not support calling message."
        raise BiNodeException(err)
    
    def stop_message(self, msg=None):
        """Receive message and return None, msg or (msg, target).
        
        If the return value is (msg, target) then stop_message(msg) is called on
        the target node.
        
        This template method calls the _stop_message method.
        """
        if not msg:
            return self._stop_message()
        arg_keys = inspect.getargspec(self._stop_message)[0]
        msg, msg_kwargs, target = self._parse_message(msg, arg_keys)
        result = self._stop_message(**msg_kwargs)
        return self._combine_message_result(result, msg, target)

    def _stop_message(self):
        """Hook method, overwrite when needed. 
        
        This default implementation only raises an exception.
        """
        err = "This node does not support calling stop_message."
        raise BiNodeException(err)
    
    def global_message(self, msg=None):
        """Receive a message and return None.
        
        Note that keys that come with a node_id are removed from msg.
        
        This template method calls the _global_message method.
        """
        arg_keys = inspect.getargspec(self._global_message)[0]
        msg_kwargs = self._parse_global_message(msg, arg_keys)
        self._global_message(**msg_kwargs)
    
    def _global_message(self):
        """Hook method, overwrite when needed."""
        pass
    
    ## Additional new methods. ##
    
    @property
    def node_id(self):
        """Return the node id (should be string) or None."""
        return self._node_id
        
    def bi_reset(self):
        """Reset the node for the next data chunck.
        
        This method is automatically called by BiFlow after the processing of
        a data chunk is completed (during both training and execution).
        
        All temporary data should be deleted. The internal node structure can
        be reset for the next data chunk. This is especially important if this
        node is called multiple times for a single chunk and an internal state
        keeps track of the actions to be performed for each call.
        """
        pass
    
    def is_bi_training(self):
        """Return True if a node is currently in a data gathering state.
        
        This method should return True if the node is gathering any data which
        is internally stored beyond the bi_reset() call. The typical example is
        some kind of learning in execute or message calls. 
        Changes to the node by stop_message alone do not fall into this
        category. 
        
        This method is used by the parallel package to decide if a node has to
        to be forked or if a copy is used.
        Basically you should ask yourself if it is sufficient to perform
        execute or message on a copy of this Node that is later discarded, or 
        if any data gathered during the operation should be stored. 
        """
        return False
    
    def _request_node_id(self, node_id):
        """Return the node if it matches the provided node id.
        
        Otherwise the return value is None. In this default implementation
        self is returned if node_id == self._node_id.
        
        Always use this method instead of directly accessing self._node_id.
        This allows a node to be associated with multiple node_ids. Otherwise
        node_ids would not work for container nodes like BiFlowNode.
        """
        if self._node_id == node_id:
            return self
        else:
            return None
        
    ### Helper methods for msg handling. ###
    
    def _parse_message(self, msg, arg_keys):
        """Return the modified message, msg kwargs and target for this node.
        
        So the return value is the tuple (msg, msg_kwargs, target). 
        The searching for a 'target' key happens after the arg_keys have been
        extracted.
        
        msg -- The original msg dict (note that this is also modified
            in-place).
        arg_keys -- List of keyword argument strings of the function the
            parsing is done for.
        
        Note that keys that come with a node_id are removed from msg (in place).
        """
        # note that arg_keys also contains 'self' (and for _execute 'x'),
        # but this should not be a problem, so there is no need to remove them
        msg_kwargs = {}
        # collect general keys
        msg_kwargs.update(dict([(key, msg[key])
                                for key in msg if key in arg_keys]))
        # collect node_id specific keys and remove them from the msg
        id_key_pairs = [(key.split(NODE_ID_KEY), key)
                        for key in msg if key.find(NODE_ID_KEY) > 0]
        msg_kwargs.update(dict([(key, msg.pop(fullkey))  # notice remove by pop
                                for (node_id, key), fullkey in id_key_pairs
                                if ((key in arg_keys) and 
                                    self._request_node_id(node_id))]))
        # deal with special keywords
        # search for target value
        target = None
        if msg and (TARGET_KEY not in msg_kwargs):
            if TARGET_KEY in msg:
                target = msg[TARGET_KEY]
            for (node_id, key), fullkey in id_key_pairs:
                if (key == TARGET_KEY) and self._request_node_id(node_id):
                    target = msg.pop(fullkey)
                    break
        # look for msg argument 
        if "msg" in arg_keys:
            msg_kwargs["msg"] = msg
            msg = None
        return msg, msg_kwargs, target
    
    def _parse_global_message(self, msg, arg_keys):
        """Return the global msg kwargs for this node and modify msg if needed.
        
        msg -- The original msg dict (note that this might get modified
            in-place).
        arg_keys -- List of keyword argument strings of the function the
            parsing is done for. Unlike in _parse_message the 'msg' key is not
            supported. 
        
        Note that keys that come with a node_id are removed from msg.
        """
        msg_kwargs = {}
        # collect general global keys
        msg_kwargs.update(dict([(key[1:], msg[key])
                                for key in msg 
                                if (key.startswith(GLOBAL_CHAR) and 
                                    key[1:] in arg_keys)]))
        # collect global node_id keys and remove them from the msg
        id_key_pairs = [(key.split(GLOBAL_ID_KEY), key)
                        for key in msg if key.find(GLOBAL_ID_KEY) > 0]
        msg_kwargs.update(dict([(key, msg.pop(fullkey))  # notice remove by pop
                                for (node_id, key), fullkey in id_key_pairs
                                if ((key in arg_keys) and 
                                    self._request_node_id(node_id))]))
        if "msg" in arg_keys:
            msg_kwargs["msg"] = msg
        return msg_kwargs
    
    def _combine_message_result(self, result, msg, target):
        """Combine the result value with the msg and the target.

        result -- Result of the form None, msg or (msg, target).
        msg -- Remaining msg after parsing.
        target -- Target value originally extracted from msg (can be None).
        
        The return value is the updated result.
        """
        # update message dict in result
        if not result:
            result = msg
        elif msg:
            if not isinstance(result, tuple):
                # result is msg
                msg.update(result)
                result = msg
            else:
                if result[0]:
                    msg.update(result[0])
                result = (msg, result[1])
        # update target value in result
        if target is not None:
            # replace target result value if necessary
            if isinstance(result, tuple):
                if result[1] is None:
                    result = (result[0], target)
            else:
                result = (result, target)
        return result
    
