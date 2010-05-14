from __future__ import absolute_import

from ..binode import BiNode, MSG_ID_SEP


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


class SenderBiNode(IdentityBiNode):
    """Sends the incoming x data to another node via bi_message."""
    
    def __init__(self, recipient_id=None, node_id=None, input_dim=None,
                 dtype=None):
        """Initialize the internal variables.
        
        recipient_id -- None or the id for the data recipient.
        """
        super(SenderBiNode, self).__init__(node_id=node_id,
                                           input_dim=input_dim,
                                           output_dim=input_dim,
                                           dtype=dtype)
        self._recipient_id = recipient_id
        
    def _execute(self, x, no_x=None):
        """Add x to the message (adressed to a specific target if defined)."""
        msg = dict()
        if self._recipient_id:
            msg[self._recipient_id + MSG_ID_SEP + "msg_x"] = x
        else:
            msg["msg_x"] = x
        if no_x:
            x = None
        else:
            return x, msg
    
