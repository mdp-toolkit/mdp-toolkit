from builtins import range

import mdp
import mdp.hinet as hinet
n = mdp.numx

from bimdp import BiNode, BiNodeException


class CloneBiLayerException(BiNodeException):
    """CloneBiLayer specific exception."""
    pass


class CloneBiLayer(BiNode, hinet.CloneLayer):
    """BiMDP version of CloneLayer.

    Since all the nodes in the layer are identical, it is guaranteed that the
    target identities match. The outgoing data on the other hand is not checked.
    So if the notes return different kinds of results the overall result is very
    unpredictable.

    The incoming data is split into len(self.nodes) parts, so the actual chunk
    size does not matter as long as it is compatible with this scheme.
    This also means that this class can deal with incoming data from a
    BiSwitchboard that is being send down.
    Arrays in the message are split up if they can be evenlty split into
    len(self.nodes) parts along the second axis, otherwise they are put
    into each node message. Arrays in the outgoing message are joined along
    the second axis (unless they are the same unsplit array), so if an array
    is accidently split no harm should be done (there is only some overhead).

    Note that a msg is always passed to the internal nodes, even if the Layer
    itself was targeted. Additional target resolution can then happen in the
    internal node (e.g. like it is done in the standard BiFlowNode).
    
    Both incomming and outgoing messages are automatically checked for the
    the use_copies msg key.
    """

    def __init__(self, node, n_nodes=1, use_copies=False,
                 node_id=None, dtype=None):
        """Initialize the internal variables.

        node -- Node which makes up the layer.
        n_nodes -- Number of times the node is repeated in this layer.
        use_copies -- Determines if a single instance or copies of the node
            are used.
        """
        super(CloneBiLayer, self).__init__(node_id=node_id, node=node,
                                           n_nodes=n_nodes, dtype=dtype)
        # (self.node is None) is used as flag for self.use_copies
        self.use_copies = use_copies

    use_copies = property(fget=lambda self: self._get_use_copies(),
                          fset=lambda self, flag: self._set_use_copies(flag))

    def _get_use_copies(self):
        """Return the use_copies flag."""
        return self.node is None

    def _set_use_copies(self, use_copies):
        """Switch internally between using a single node instance or copies.

        In a normal CloneLayer a single node instance is used to represent all
        the horizontally aligned nodes. But in a BiMDP where the nodes store
        temporary data this may not work.
        Via this method one can therefore create copies of the single node
        instance.

        This method can also be triggered by the use_copies msg key.
        """
        if use_copies and (self.node is not None):
            # switch to node copies
            self.nodes = [self.node.copy() for _ in range(len(self.nodes))]
            self.node = None  # disable single node while copies are used
        elif (not use_copies) and (self.node is None):
            # switch to a single node instance
            if self.is_training():
                err = ("Calling switch_to_instance during training will "
                       "probably result in lost training data.")
                raise CloneBiLayerException(err)
            elif self.is_bi_training():
                err = ("Calling switch_to_instance during bi_learning will "
                       "probably result in lost learning data.")
                raise CloneBiLayerException(err)
            self.node = self.nodes[0]
            self.nodes = [self.node] * len(self.nodes)

    def _get_method(self, method_name, default_method, target):
        """Return the default method and the unaltered target.

        This method overrides the standard BiNode _get_method to delegate the
        method selection to the internal nodes.
        """
        return default_method, target

    ## standard node methods ##

    def _check_input(self, x):
        """Input check is disabled.

        It will be checked by the targeted internal node.
        """
        pass

    def _execute(self, x, msg=None):
        """Process the data through the internal nodes."""
        if msg is not None:
            self._extract_message_copy_flag(msg)
        y_results = []
        msg_results = []
        target = None
        node_msgs = self._get_split_messages(msg)
        if x is not None:
            # use the dimension of x, because this also works for inverse
            node_dim = x.shape[1] // len(self.nodes)
        else:
            node_dim = None
        for i_node, node in enumerate(self.nodes):
            if node_dim:
                node_x = x[:, node_dim*i_node : node_dim*(i_node+1)]
            else:
                node_x = None
            node_msg = node_msgs[i_node]
            if node_msg:
                node_result = node.execute(node_x, node_msg)
            else:
                node_result = node.execute(node_x)
            ## store result
            if not isinstance(node_result, tuple):
                y_results.append(node_result)
            else:
                y_results.append(node_result[0])
                msg_results.append(node_result[1])
                if len(node_result) == 3:
                    target = node_result[2]
        ## combine message results
        msg = self._get_combined_message(msg_results)
        if (not y_results) or (y_results[-1] is None):
            y = None
        else:
            y = n.hstack(y_results)
        # check outgoing message for use_copies key
        if msg is not None:
            self._extract_message_copy_flag(msg)
        ## return result
        if target is not None:
            return (y, msg, target)
        elif msg:
            return (y, msg)
        else:
            return y

    def _train(self, x, msg=None):
        """Perform single training step by training the internal nodes."""
        ## this code is mostly identical to the execute code,
        ## currently the only difference is that train is called
        if msg is not None:
            self._extract_message_copy_flag(msg)
        y_results = []
        msg_results = []
        target = None
        node_msgs = self._get_split_messages(msg)
        if x is not None:
            # use the dimension of x, because this also works for inverse
            node_dim = x.shape[1] // len(self.nodes)
        else:
            node_dim = None
        for i_node, node in enumerate(self.nodes):
            if node_dim:
                node_x = x[:, node_dim*i_node : node_dim*(i_node+1)]
            else:
                node_x = None
            node_msg = node_msgs[i_node]
            if node_msg:
                node_result = node.train(node_x, node_msg)
            else:
                node_result = node.train(node_x)
            ## store result
            if not isinstance(node_result, tuple):
                y_results.append(node_result)
            else:
                y_results.append(node_result[0])
                msg_results.append(node_result[1])
                if len(node_result) == 3:
                    target = node_result[2]
        ## combine message results
        msg = self._get_combined_message(msg_results)
        if (not y_results) or (y_results[-1] is None):
            y = None
        else:
            y = n.hstack(y_results)
        # check outgoing message for use_copies key
        if msg is not None:
            self._extract_message_copy_flag(msg)
        ## return result
        if target is not None:
            return (y, msg, target)
        elif msg:
            return (y, msg)
        else:
            return y

    def _stop_training(self, msg=None):
        """Call stop_training on the internal nodes.

        The outgoing result message is also searched for a use_copies key,
        which is then applied if found.
        """
        if msg is not None:
            self._extract_message_copy_flag(msg)
        target = None
        if self.use_copies:
            ## have to call stop_training for each node
            y_results = []
            msg_results = []
            node_msgs = self._get_split_messages(msg)
            for i_node, node in enumerate(self.nodes):
                node_msg = node_msgs[i_node]
                if node_msg:
                    node_result = node.stop_training(node_msg)
                else:
                    node_result = node.stop_training()
                ## store result
                if not isinstance(node_result, tuple):
                    y_results.append(node_result)
                else:
                    y_results.append(node_result[0])
                    msg_results.append(node_result[1])
                    if len(node_result) == 3:
                        target = node_result[2]
            ## combine message results
            msg = self._get_combined_message(msg_results)
            if (not y_results) or (y_results[-1] is None):
                y = None
            else:
                y = n.hstack(y_results)
        else:  
            ## simple case of a single instance
            node_result = self.node.stop_training(msg)
            if not isinstance(node_result, tuple):
                return node_result
            elif len(node_result) == 2:
                y, msg = node_result
            else:
                y, msg, target = node_result
        # check outgoing message for use_copies key
        if msg is not None:
            self._extract_message_copy_flag(msg)
        # return result
        if target is not None:
            return (y, msg, target)
        elif msg:
            return (y, msg)
        else:
            return y

    ## BiNode methods ##

    def _bi_reset(self):
        """Call bi_reset on all the inner nodes."""
        if self.use_copies:
            for node in self.nodes:
                node.bi_reset()
        else:
            # note: reaching this code probably means that copies should be used
            self.node.bi_reset()

    def _request_node_id(self, node_id):
        """Return an internal node if it matches the provided node id.

        If the node_id matches that of the layer itself, then self is returned.
        """
        if self.node_id == node_id:
            return self
        if not self.use_copies:
            return self.node._request_node_id(node_id)
        else:
            # return the first find, but call _request_node_id on all copies
            # otherwise BiFlowNode._last_id_request would get confused
            first_found_node = None
            for node in self.nodes:
                found_node = node._request_node_id(node_id)
                if (not first_found_node) and found_node:
                    first_found_node = found_node
            return first_found_node

    ## Helper methods for message handling ##

    def _extract_message_copy_flag(self, msg):
        """Look for the the possible copy flag and modify the msg if needed.

        If the copy flag is found the Node is switched accordingly.
        """
        msg_id_keys = self._get_msg_id_keys(msg)
        copy_flag = self._extract_message_key("use_copies", msg, msg_id_keys)
        if copy_flag is not None:
            self.use_copies = copy_flag

    def _get_split_messages(self, msg):
        """Return messages for the individual nodes."""
        if not msg:
            return [None] * len(self.nodes)
        msgs = [dict() for _ in range(len(self.nodes))]
        n_nodes = len(self.nodes)
        for (key, value) in list(msg.items()):
            if (isinstance(value, n.ndarray) and
                # check if the array can be split up
                len(value.shape) >= 2 and not value.shape[1] % n_nodes):
                # split the data along the second index
                split_values = n.hsplit(value, n_nodes)
                for i, split_value in enumerate(split_values):
                    msgs[i][key] = split_value
            else:
                for node_msg in msgs:
                    # Note: the value is not copied, just referenced
                    node_msg[key] = value
        return msgs

    def _get_combined_message(self, msgs):
        """Return the combined message.

        Only keys from the last entry in msgs are used. Only when the value
        is an array are all the msg values combined.
        """
        if (not msgs) or (msgs[-1] is None):
            return None
        if len(msgs) == 1:
            return msgs[0]
        msg = dict()
        for (key, one_value) in list(msgs[-1].items()):
            other_value = msgs[0][key]
            if (isinstance(one_value, n.ndarray) and
                # check if the array was originally split up
                (len(one_value.shape) >= 2 and one_value is not other_value)):
                msg[key] = n.hstack([node_msg[key] for node_msg in msgs])
            else:
                # pick the msg value of the last node
                msg[key] = msgs[-1][key]
        return msg
