
from bimdp import BiNode, MSG_ID_SEP
from bimdp.nodes import IdentityBiNode


class SenderBiNode(IdentityBiNode):
    """Sends the incoming x data to another node via bi_message."""

    def __init__(self, recipient_id=None, **kwargs):
        """Initialize the internal variables.

        recipient_id -- None or the id for the data recipient.
        """
        super(SenderBiNode, self).__init__(**kwargs)
        self._recipient_id = recipient_id

    def _execute(self, x, no_x=None):
        """Add msg_x to the message (adressed to a target if defined)."""
        msg = dict()
        if self._recipient_id:
            msg[self._recipient_id + MSG_ID_SEP + "msg_x"] = x
        else:
            msg["msg_x"] = x
        if no_x:
            x = None
        return x, msg
