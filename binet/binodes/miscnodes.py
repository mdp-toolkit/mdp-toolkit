from __future__ import absolute_import

from ..binode import BiNode, NODE_ID_KEY


class IdentityBiNode(BiNode):
    """Identity implementation for a BiNode.
    
    The arguments are simply passed through in execute.
    
    Instances of this class can be used as simple jump targets in a flow.
    """
    
    def _execute(self, x):
        """Return x and msg."""
        return x
    
    def _set_input_dim(self, n):
        """If input dim is given set output dim to the same value."""
        self._input_dim = n
        self.set_output_dim(n)
    
    def is_trainable(self):
        return False
    
    def is_bi_training(self):
        return False


### Classes to take care of the target behavior. ###

class JumpBiNode(IdentityBiNode):
    """BiNode which can perform all kinds of jumps.
    
    This is useful for testing or flow control. It can also be used 
    together with BiNodes as simple jump targets.
    """
    
    # TODO: implement multiple train results, when execution is continued
    
    def __init__(self, train_results=None, stop_train_results=None, 
                 execute_results=None, message_results=None,
                 stop_message_results=None, *args, **kwargs):
        """Initialize this BiNode.
        
        Note that this node has an internal variable self.loop_counter which is
        used by execute, message and stop_message (and incremented by each).
        
        train_results -- List of results for the training phases. Each list
            entry can be a single result or a list (when the execution is
            continued after calling train).
        stop_train_results -- List of results for the training phases.
        execute_results -- Single result tuple starting at msg or list of
            results, which are used according to the loop counter. The list
            entries can also be None (then x is simply forwarded).
        message_results, stop_message_results -- Like execute_results.
        """
        self.loop_counter = 0 # counter for execution phase
        self._train_results = train_results
        self._stop_train_results = stop_train_results
        self._execute_results = execute_results
        self._message_results = message_results
        self._stop_message_results = stop_message_results
        super(JumpBiNode, self).__init__(*args, **kwargs)
        
    def is_trainable(self):
        if self._train_results:
            return True
        else:
            return False
    
    def is_invertible(self):
        return False
        
    def _get_train_seq(self):
        """Return a train_seq which returns the predefined values."""
        # wrapper function for _train, using local scopes
        def get_train_function(i_phase):
            def train_function(x):
                return self._train_results[i_phase]
            return train_function
        # wrapper function for _stop_training
        def get_stop_training(i_phase):
            def stop_training():
                return self._stop_train_results[i_phase]
            return stop_training
        # now wrap the training sequence
        train_seq = []
        if not self._train_results:
            return train_seq
        for i_phase in range(len(self._train_results)):
            train_seq.append((get_train_function(i_phase),
                              get_stop_training(i_phase)))
        return train_seq
    
    def _execute(self, x):
        """Return the predefined values for the current loop count value."""
        self.loop_counter += 1
        if not self._execute_results:
            return x
        if self.loop_counter-1 >= len(self._execute_results):
            return x
        result = self._execute_results[self.loop_counter-1]
        if result is None:
            return x
        else:
            return (x,) + result
            
    def _message(self):
        """Return the predefined values for the current loop count value."""
        self.loop_counter += 1
        if not self._message_results:
            return None
        if self.loop_counter-1 >= len(self._message_results):
            return None
        return self._message_results[self.loop_counter-1]
    
    def _stop_message(self):
        """Return the predefined values for the current loop count value."""
        self.loop_counter += 1
        if not self._stop_message_results:
            return None
        if self.loop_counter-1 >= len(self._stop_message_results):
            return None
        return self._stop_message_results[self.loop_counter-1]
    
    def bi_reset(self):
        """Reset the loop counter."""
        self.loop_counter = 0

    def is_bi_learning(self):
        return False
    

### Some Standard BiNodes ###

class SenderBiNode(IdentityBiNode):
    """Sends the incoming x data to another node via bi_message."""
    
    def __init__(self, target=None, msg_supplement=None, **kwargs):
        """Initialize the internal variables.
        
        target -- None or the sender target. Data will only be send if the
            target is not none.
        msg_supplement -- None or a dict that will be combined with the output
            msg.
        
        args and kwargs are forwarded via super to the next __init__ method
        in the MRO.
        """
        super(SenderBiNode, self).__init__(**kwargs)
        self._target = target
        self._msg_supplement = msg_supplement
    
    def _global_message(self, target=False, msg_supplement=False):
        """Overwrite the saved target and msg parameters.
        
        target -- Update the target.
        msg_supplement -- Update the msg_supplement.
        """
        if target is not False:
            self._target = target
        if msg_supplement is not False:
            self._msg_supplement = msg_supplement
                
    def _execute(self, x, msg=None):
        """Send the x value to the stored target via bi_message.
        
        The x in the message has the key bi_x. 
        """
        if self._target is not None:
            if isinstance(self._target, str):
                bi_msg = {self._target + NODE_ID_KEY + "x": x}
            else:
                bi_msg = {"x": x}
            if self._msg_supplement is not None:
                bi_msg.update(self._msg_supplement)
            return x, msg, None, bi_msg, self._target
        else:
            return x, msg