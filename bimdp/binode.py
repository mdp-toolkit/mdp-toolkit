"""
Special BiNode class derived from Node to allow complicated flow patterns.

Messages:
=========

The message argument 'msg' of the outer method 'execute' or 'train' is either a
dict or None (which is treated equivalently to an empty dict and is the default
value). The message is automatically parsed against the method signature of
_train or _execute (or any other specified method) in the following way:

    normal key string -- Is copied if in signature and passed as a named
        argument.
    
    node_id->key -- Is extracted (i.e. removed in original message) and passed 
        as a named argument. The separator '->' is also stored available
        as the constant NODE_ID_KEY. 
        
The msg returned from the inner part of the method (e.g. _execute) is then used
to update the original message (so values can be overwritten).

If the message results in passing kwargs that are not args of the Node or if
args without default value are missing in the message, this will result in the
standard Python missing-arguments-exception (this is not checked by BiNode
itself). 


BiNode Return Value Options:
============================

 result for execute:
     x, (x, msg), (x, msg, target)
     
 result for train:
    None -- terminates training
    x, (x, msg), (x, msg, target) -- Execution is continued and
        this node will be reached at a later time to terminate training.
        If the result has the form (None, msg) then the msg is dropped (so
        it is not required to 'clear' the message manually).
     
 result for stop_training and stop_message:
    None -- terminates the stop_message propagation
    (msg, target) -- If no target is specified then the remaining msg is
        dropped (terminates the propagation).
    

Magic message keys:
===================

When the incoming message is parsed by the BiNode base class, some argument
keywords are treated in a special way:

 'msg' -- If any method like _execute accept a 'msg' keyword then the complete
     remaining message (after parsing the other keywords) is supplied. The
     message in the return value then completely replaces the original message
     (instead of only updating it). This way a node can completely control the
     message and for example remove keys.
     
 'target' -- If any template method like execute finds a 'target' keyword in
     the message then this is used as the target value in the return value.
     However, if _execute then also returns a target value this overwrites the
     target value. In global_message calls 'target' has no special meaning and
     can be used like any other keyword.
     
  'method' -- Specify the name of the method that should be used instead of
      the standard one (e.g. in execute the standard method is _execute).
      An underscore is automatically added in front, so to select _execute
      one would have to provide 'execute'.
      If 'inverse' is given then the inverse dimension check will be performed
      and if no target is provided it will be set to -1.
      
  'method' in stop_message -- To make it possible to call 'execute' and
      'inverse' in 'stop_message' there is some magic going on if these are
      specified as message arguments: In addition to the normal automatic
      extraction of an 'x' key from the message the array output of the node
      is also stored back as 'x' in the message (overwriting the previous
      value). Additionally the target is given a default value of 1 or -1
      (so setting the 'method' value is sufficient for normal execution
      or inverse).
     
"""

import inspect

import mdp

# separator for node_id in message keys
MSG_ID_SEP = "->"

# methods that can overwrite docs:
if '_stop_message' not in mdp.NodeMetaclass.DOC_METHODS:
    mdp.NodeMetaclass.DOC_METHODS.append('_stop_message')


class BiNodeException(mdp.NodeException):
    """Exception for BiNode problems."""
    pass

    
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
    
    def __init__(self, node_id=None, stop_result=None, **kwargs):
        """Initialize BiNode.
        
        node_id -- None or string which identifies the node.
        stop_result -- A (msg, target) tupple which is used as the result for
            stop_training (but can be overwritten by any actual results).
            If the node has multiple training phases then this must be None or
            an iterable with one entry for each training phase.
        
        kwargs are forwarded via super to the next __init__ method
        in the MRO.
        """
        self._node_id = node_id
        self._stop_result = stop_result
        super(BiNode, self).__init__(**kwargs)
        
    ### Modified template methods from mdp.Node. ###
    
    def execute(self, x, msg=None):
        """Return single value y or a result tuple.
        
        x can be None, then the usual checks are omitted.

        The possible return types are y, (y, msg), (y, msg, target)
        The outgoing msg carries forward the incoming message content. 
        The last entry in a result tuple must not be None. 
        y can be None if the result is a tuple.
        
        This template method normally calls the corresponding _execute method
        or another method as specified in the message (using the magic 'method'
        key. 
        """
        if msg is None:
            return super(BiNode, self).execute(x)
        msg_id_keys = self._get_msg_id_keys(msg)
        target = self._extract_message_key("target", msg, msg_id_keys)
        method_name = self._extract_message_key("method", msg, msg_id_keys)
        method, target = self._get_method(method_name, self._execute, target)
        msg, arg_dict = self._extract_method_args(method, msg, msg_id_keys)
        # perform specific checks
        if x is not None:
            if (not method_name) or (method_name == "execute"):
                self._pre_execution_checks(x)
                x = self._refcast(x)
            # testing for the actual method allows nodes to delegate method
            # resolution to internal nodes by manipulating _get_method
            elif method == self._inverse:
                self._pre_inversion_checks(x)
        result = method(x, **arg_dict)
        return self._combine_execute_result(result, msg, target)
    
    def is_trainable(self):
        """Return the return value from super."""
        return super(BiNode, self).is_trainable()
    
    def train(self, x, msg=None):
        """Train and return None or more if the execution should continue.
        
        The possible return types are None, y, (y, msg), (y, msg, target).
        The last entry in a result tuple must not be None.
        y can be None if the result is a tuple.
        
        This template method normally calls the corresponding _train method
        or another method as specified in the message (using the magic 'method'
        key. 

        Note that the remaining msg and taret values are only used if _train
        (or the requested method) returns something different from None
        (so an empty dict can be used to trigger continued execution).
        """
        # perform checks, adapted from Node.train
        if not self.is_trainable():
            raise mdp.IsNotTrainableException("This node is not trainable.")
        if not self.is_training():
            err = "The training phase has already finished."
            raise mdp.TrainingFinishedException(err)
        if msg is None:
            # no fall-back on Node.train because we might have a return value
            self._check_input(x)
            self._check_train_args(x)        
            self._train_phase_started = True
            x = self._refcast(x)
            return self._train_seq[self._train_phase][0](x)
        msg_id_keys = self._get_msg_id_keys(msg)
        target = self._extract_message_key("target", msg, msg_id_keys)
        method_name = self._extract_message_key("method", msg, msg_id_keys)
        default_method = self._train_seq[self._train_phase][0]
        method, target = self._get_method(method_name, default_method, target)
        msg, arg_dict = self._extract_method_args(method, msg, msg_id_keys)
        # perform specific checks
        if x is not None:
            if (not method_name) or (method_name == "train"):
                self._check_input(x)
                self._check_train_args(x, **arg_dict)  
                self._train_phase_started = True
                x = self._refcast(x)
            elif method == self._inverse:
                self._pre_inversion_checks(x)
        result = method(x, **arg_dict)
        if result is None:
            return None
        result = self._combine_execute_result(result, msg, target)
        if (isinstance(result, tuple) and len(result) == 2 and
            result[0] is None):
            # drop the remaining msg, so that no maual clearing is required
            return None
        return result 
    
    def stop_training(self, msg=None):
        """Stop training phase and return None or (msg, target).
        
        The result tuple is then used to call stop_message on the target node.
        The outgoing msg carries forward the incoming message content.
        
        This template method calls a _stop_training method from self._train_seq.
        Note that it is not possible to select other methods via the 'method'
        message key.
        
        If a stop_result was given in __init__ then it is used but can be
        overwritten by the _stop_training result.
        """
        # basic checks
        if self.is_training() and self._train_phase_started == False:
            raise mdp.TrainingException("The node has not been trained.")
        if not self.is_training():
            err = "The training phase has already finished."
            raise mdp.TrainingFinishedException(err)
        # call stop_training
        if not msg:
            result = self._train_seq[self._train_phase][1]()
            target = None
        else:
            msg_id_keys = self._get_msg_id_keys(msg)
            target = self._extract_message_key("target", msg, msg_id_keys)
            method_name = self._extract_message_key("method", msg, msg_id_keys)
            default_method = self._train_seq[self._train_phase][1]
            method, target = self._get_method(method_name,
                                              default_method, target)
            msg, arg_dict = self._extract_method_args(method, msg, msg_id_keys)
            result = method(**arg_dict)
        # close the current phase
        self._train_phase += 1
        self._train_phase_started = False
        # check if we have some training phase left
        if self.get_remaining_train_phase() == 0:
            self._training = False
        # use stored stop message and update it with the result
        if self._stop_result:
            if self.has_multiple_training_phases():
                stored_stop_result = self._stop_result[self._train_phase - 1]
            else:
                stored_stop_result = self._stop_result
            # make sure that the original dict in stored_stop_result is not
            # modified (this could have unexpected consequences in some cases)
            stored_msg = stored_stop_result[0].copy()
            if msg:
                stored_msg.update(msg)
            msg = stored_msg
            if target is None:
                target = stored_stop_result[1]
        return self._combine_stop_message_result(result, msg, target)
    
    ### New methods for node messaging. ###
    
    def stop_message(self, msg=None):
        """Receive message and return None, msg or (msg, target).
        
        If the return value is (msg, target) then stop_message(msg) is called
        on the target node.
        
        This template method calls the _stop_message method or another method
        specified in the method. If the specified method is specified as
        'execute' or 'inverse' then the x/y return value is extracted and
        stored in the message under the key 'x' and the target defaults to
        1 or -1 (so setting the 'method' value is sufficient for normal
        execution or inverse).
        """
        if not msg:
            return self._stop_message()
        msg_id_keys = self._get_msg_id_keys(msg)
        target = self._extract_message_key("target", msg, msg_id_keys)
        method_name = self._extract_message_key("method", msg, msg_id_keys)
        method, target = self._get_method(method_name, self._stop_message,
                                          target)
        if method_name == "execute" and target is None:
            # use 1 as default target for execution, note that in order to
            # terminate the execution one therefore has to call another method
            target = 1
        # TODO: perform dimensionality checks for execute/inverse?
        msg, arg_dict = self._extract_method_args(method, msg, msg_id_keys)
        result = method(**arg_dict)
        if method_name == "execute" or method_name == "inverse":
            ## magic to add array result back to message
            if result is not None:
                if isinstance(result, tuple):
                    x = result[0]
                    if result[1] is not None:
                        # result[1] must be dict
                        result[1]["x"] = x
                        if len(result) > 2:
                            result = result[1:]
                        else:
                            result = result[1]
                    else:
                        if len(result) > 2:
                            result = ({"x": x}, result[2])
                        else:
                            result = {"x": x}
                else:
                    # result must be single x value
                    result = {"x": result}
        return self._combine_stop_message_result(result, msg, target)
    
    def _stop_message(self):
        """Hook method, overwrite when needed. 
        
        This default implementation only raises an exception.
        """
        err = ("This node (%s) does not support calling stop_message." %
               str(self))
        raise BiNodeException(err)
    
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
        
        Use this method instead of directly accessing self._node_id.
        This allows a node to be associated with multiple node_ids. Otherwise
        node_ids would not work for container nodes like BiFlowNode.
        """
        if self._node_id == node_id:
            return self
        else:
            return None
        
    ### Helper methods for msg handling. ###
    
    def _get_msg_id_keys(self, msg):
        """Return the id specific message keys for this node.
        
        The format is [(key, fullkey),...].
        """
        msg_id_keys = []
        for fullkey in msg:
            if fullkey.find(MSG_ID_SEP) > 0:
                node_id, key = fullkey.split(MSG_ID_SEP)
                if node_id == self._node_id:
                    msg_id_keys.append((key, fullkey))
        return msg_id_keys
                    
    @staticmethod
    def _extract_message_key(key, msg, msg_id_keys):
        """Extract and return the requested key from the message.

        Note that msg is modfied if the found key was node_id specific.
        """
        value = None
        if key in msg:
            value = msg[key]
        # check for node_id specific key and remove it from the msg
        for _key, _fullkey in msg_id_keys:
            if key == _key:
                value = msg.pop(_fullkey)
                break
        return value
    
    @staticmethod
    def _extract_method_args(method, msg, msg_id_keys):
        """Extract the method arguments form the message.
        
        Return the new message and a dict with the keyword arguments (the
        return of the message is done because it can be set to None).
        """
        arg_keys = inspect.getargspec(method)[0]   
        arg_dict = dict((key, msg[key]) for key in msg if key in arg_keys)
        arg_dict.update(dict((key, msg.pop(fullkey))  # notice remove by pop
                             for key, fullkey in msg_id_keys
                             if key in arg_keys))
        if "msg" in arg_keys:
            arg_dict["msg"] = msg
            msg = None
        return msg, arg_dict
      
    def _get_method(self, method_name, default_method, target=None):
        """Return the method to be called and the target return value.
        
        method_name -- as provided in msg (without underscore)
        default_method -- bound method object
        target -- return target value as provided in message or None
        
        If the chosen method is _inverse then the default target is -1.
        """
        if not method_name:
            method = default_method
        elif method_name == "inverse":
            method = self._inverse
            if target is None:
                target = -1
        else:
            method_name = "_" + method_name
            try:
                method = getattr(self, method_name)
            except AttributeError:
                err = ("The message requested a method named '%s', but "
                       "there is no such method." % method_name)
                raise BiNodeException(err)
        return method, target
    
    @staticmethod
    def _combine_stop_message_result(result, msg, target):
        """Combine the message result with the provided values.
        
        result -- None, msg or (msg, target)
        
        The values in result always have priority.
        If not target is available then the remaining message is dropped.
        """
        if not result:
            if target is not None:
                return (msg, target)
            else:
                return None
        elif not isinstance(result, tuple):
            # result has not target, terminate stop_message propagation
            if target is None:
                return None
            if msg:
                msg.update(result)
            else:
                msg = result
            return (msg, target)
        else:
            # result contains target value 
            if msg:
                if result[0]:
                    msg.update(result[0])
                return msg, result[1]
            else:
                return result
                
    @staticmethod
    def _combine_execute_result(result, msg, target):
        """Combine the execution result with the provided values.
        
        result -- x, (x, msg) or (x, msg, target)
        
        The values in result always has priority.
        """
        # overwrite result values if necessary and return
        if isinstance(result, tuple):
            if msg:
                if result[1]:
                    # combine outgoing msg and remaining msg values
                    msg.update(result[1])
                result = (result[0], msg) + result[2:]
            if (target is not None) and (len(result) == 2):
                # use given target if no target value was returned
                result += (target,)
            return result
        else:
            # result is only single array
            if (not msg) and (target is None):
                return result
            elif target is None:
                return result, msg
            else:
                return result, msg, target
    
    ### Overwrite Special Methods ###
        
    def __repr__(self):
        """BiNode version of the Node representation, adding the node_id.""" 
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        if self.dtype is None:
            typ = 'dtype=None'
        else:
            typ = "dtype='%s'" % self.dtype.name
        node_id = self.node_id
        if node_id is None:
            nid = 'node_id=None'
        else:
            nid = 'node_id="%s"' % node_id
        args = ', '.join((inp, out, typ, nid))
        return name + '(' + args + ')'
    
    def __add__(self, other):
        """Adding binodes returns a BiFlow.
        
        If a normal Node or Flow is added to a BiNode then a BiFlow is
        returned.
        Note that if a flow is added then a deep copy is used (deep
        copies of the nodes are used).
        """
        # unfortunately the inline import are required to avoid
        # a cyclic import (unless one add a helper function somewhere else)
        if isinstance(other, mdp.Node):
            import bimdp
            return bimdp.BiFlow([self, other])
        elif isinstance(other, mdp.Flow):
            flow_copy = other.copy()
            import bimdp
            biflow = bimdp.BiFlow([self.copy()] + flow_copy.flow)
            return biflow
        else:
            # can delegate old cases
            return super(BiNode, self).__add__(other)
        
        