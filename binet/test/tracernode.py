"""
Classes for tracing BiNode behavior in flows.
"""

import mdp

import binet


class TraceJumpBiNode(binet.JumpBiNode):
    """Node for testing, that logs when and how it is called."""
    
    def __init__(self, tracelog, log_data=False, verbose=False,
                 *args, **kwargs):
        """Initialize the node.
        
        tracelog -- list to which to append the log entries
        log_data -- if true the data will be logged as well
        """
        self._tracelog = tracelog
        self._log_data = log_data
        self._verbose = verbose 
        super(TraceJumpBiNode, self).__init__(*args, **kwargs)
        
    def train(self, x, msg=None):
        if self._log_data:
            self._tracelog.append((self._node_id, "train", x, msg))
        else:
            self._tracelog.append((self._node_id, "train"))
        if self._verbose:
            print self._tracelog[-1]
        return super(TraceJumpBiNode, self).train(x, msg)
        
    def execute(self, x, msg):
        if self._log_data:
            self._tracelog.append((self._node_id, "execute", x, msg))
        else:
            self._tracelog.append((self._node_id, "execute"))
        if self._verbose:
            print self._tracelog[-1]
        return super(TraceJumpBiNode, self).execute(x, msg)
        
    def stop_training(self):
        self._tracelog.append((self._node_id, "stop_training"))
        if self._verbose:
            print self._tracelog[-1]
        return super(TraceJumpBiNode, self).stop_training()
        
    def stop_message(self, msg=None):
        if self._log_data:
            self._tracelog.append((self._node_id, "stop_message", msg))
        else:
            self._tracelog.append((self._node_id, "stop_message"))
        if self._verbose:
            print self._tracelog[-1]
        return super(TraceJumpBiNode, self).stop_message(msg)
        
    def bi_reset(self):
        self._tracelog.append((self._node_id, "bi_reset"))
        if self._verbose:
            print self._tracelog[-1]
        return super(TraceJumpBiNode, self).bi_reset()
    

class ParallelTraceJumpBiNode(TraceJumpBiNode):
    
    def _fork(self):
        return self.copy()
    
    def _join(self):
        pass
    
    def _bi_fork(self):
        return self.copy()
    
    def _bi_join(self):
        pass
    

class IdNode(mdp.Node):
    """Non-bi identity node for testing."""
    
    def is_trainable(self):
        return False