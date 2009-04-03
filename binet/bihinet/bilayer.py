
import mdp
import mdp.hinet as hinet
n = mdp.numx

from ..binode import BiNode, BiNodeException

# TODO: not sure if always splitting up the msg parts is always such a great
#    idea, maybe a switch should be introduced?


class CloneBiLayerException(BiNodeException):
    """CloneBiLayer specific exception."""
    pass


class CloneBiLayer(BiNode, hinet.CloneLayer):
    """BiNet version of CloneLayer.
    
    Since all the nodes in the layer are identical, it is guaranteed that the
    target identities match. The outgoing data on the other hand is not checked.
    So if the notes return different kinds of results the overall result is very
    unpredictable.
    
    The incoming data is split into len(self.nodes) chunks, so the actual chunk
    size does not matter as long as it is compatible with this scheme.
    This also means that this class can deal with incoming data from a
    BiSwitchboard that is being send down.
    
    Note that a msg is always passed to the internal nodes, even if the Layer
    itself was targeted. Additional target resolution can then happen in the
    internal node (e.g. like it is done in the standard BiFlowNode).
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
        the horizontally aligned nodes. But in a BiNet where the nodes store 
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
            self.nodes = (self.node,) * len(self.nodes) 
         
    ## standard node methods ##
    
    def _check_input(self, x):
        """Input check is disabled.
        
        It will be checked by the targeted internal node.
        """  
        pass
    
    def _execute(self, x, msg=None):
        """Process the data through the internal nodes."""
        x_start_index = 0
        x_stop_index = 0
        y_results = []
        msgs = []
        target = None
        branch_msgs = []
        branch_target = None
        for (node, node_msg) in zip(self.nodes, self._get_split_msgs(msg)):
            if x is not None:
                x_start_index = x_stop_index
                x_stop_index += node.input_dim
                node_x = x[:, x_start_index : x_stop_index]
            else:
                node_x = None
            if node_msg:
                node_result = node.execute(node_x, node_msg)
            else:
                node_result = node.execute(node_x)
            ## store result
            if not isinstance(node_result, tuple):
                y_results.append(node_result)
            else:
                y_results.append(node_result[0])
                msgs.append(node_result[1])
                if len(node_result) >= 3:
                    target = node_result[2]
                if len(node_result) >= 4:
                    branch_msgs.append(node_result[3])
                if len(node_result) == 5:
                    branch_target = node_result[4]
        ## combine message results
        msg = self._get_combined_message(msgs)
        branch_msg = self._get_combined_message(branch_msgs)
        if (not y_results) or (y_results[-1] is None):
            y = None
        else:
            y = n.hstack(y_results)
        ## return result
        if branch_target is not None:
            return (y, msg, target, branch_msg, branch_target)
        elif branch_msg:
            return (y, msg, target, branch_msg)
        elif target is not None:
            return (y, msg, target)
        elif msg:
            return (y, msg)
        else:
            return y
        
    def _train(self, x, msg=None):
        """Perform single training step by training the internal nodes."""
        x_start_index = 0  # for array variables like x and the current node
        x_stop_index = 0
        y_results = []
        msgs = []
        target = None
        branch_msgs = []
        branch_target = None
        for (node, node_msg) in zip(self.nodes, self._get_split_msgs(msg)):
            if x is not None:
                x_start_index = x_stop_index
                x_stop_index += node.input_dim
                node_x = x[:, x_start_index : x_stop_index]
            else:
                node_x = None
            if node_msg:
                node_result = node.train(node_x, node_msg)
            else:
                node_result = node.train(node_x)
            ## store result
            if node_result is None:
                continue
            elif isinstance(node_result, dict):
                msgs.append(node_result)
            elif len(node_result) == 2:
                branch_msgs.append(node_result[0])
                branch_target = node_result[1]
            elif len(node_result) >= 3:
                y_results.append(node_result[0])
                msgs.append(node_result[1])
                target = node_result[2]
                if len(node_result) >= 4:
                    branch_msgs.append(node_result[3])
                if len(node_result) == 5:
                    branch_target = node_result[4]
        ## combine message results
        msg = self._get_combined_message(msgs)
        branch_msg = self._get_combined_message(branch_msgs)
        if (not y_results) or (y_results[-1] is None):
            y = None
        else:
            y = n.hstack(y_results)
        ## return result
        if target is None:
            if branch_target is None:
                return branch_msg
            else:
                return (branch_msg, branch_target)
        else:
            if branch_target is None:
                if branch_msg:
                    return (y, msg, target, branch_msg)
                else:
                    return (y, msg, target)
            else:
                return (y, msg, target, branch_msg, branch_target)
                
        
    def _stop_training(self, msg=None):
        """Call stop_training on the internal nodes.
        
        The outgoing message is also searched for a use_copies key, which is
        then applied if found. It is not required to provide a target for this.
        """
        if not self.use_copies:
            ## simple case of a single instance
            node_result = self.node.stop_training(msg)
            if node_result is None:
                return None
            elif isinstance(node_result, dict):
                msg = node_result
                target = None
            else:
                msg, target = node_result
        else:
            ## complex case, call stop_training on all node copies 
            msgs = []
            target = None
            for (node, node_msg) in zip(self.nodes, self._get_split_msgs(msg)):
                if node_msg:
                    node_result = node.stop_training(node_msg)
                else:
                    node_result = node.stop_training()
                if node_result is None:
                    continue
                elif isinstance(node_result, dict):
                    msgs.append(node_result)
                elif len(node_result) == 2:
                    msgs.append(node_result[0])
                    target = node_result[1]
            msg = self._get_combined_message(msgs)
        # check for outgoing message for use_copies key
        if msg is not None:
            self._parse_msg_copy_flag(msg)
        if target is None:
            return msg
        else:
            return msg, target
            
    ## BiNode methods ##
        
    def _message(self, msg=None):
        """Call message on the internal nodes."""
        msgs = []
        target = None
        for (node, node_msg) in zip(self.nodes, self._get_split_msgs(msg)):
            node_result = node.message(node_msg)
            ## store result
            if isinstance(node_result, dict):
                msgs.append(node_result)
            else:
                msgs.append(node_result[0])
                target = node_result[1]
        ## combine message results
        msg = self._get_combined_message(msgs)
        if target is None:
            return msg
        elif msg:
            return msg
    
    def _stop_message(self, use_copies=None, msg=None):
        """Call stop_message on the internal nodes.
        
        The outgoing message is also searched for a use_copies key, which is
        then applied if found. It is not required to provide a target for this.
        """
        if use_copies is not None:
            self.use_copies = use_copies
        msgs = []
        target = None
        for (node, node_msg) in zip(self.nodes, self._get_split_msgs(msg)):
            node_result = node.stop_message(node_msg)
            ## store result
            if isinstance(node_result, dict):
                msgs.append(node_result)
            else:
                msgs.append(node_result[0])
                target = node_result[1]
        ## combine message results
        msg = self._get_combined_message(msgs)
        # check for outgoing message for use_copies key
        if msg is not None:
            self._parse_msg_copy_flag(msg)
        if target is None:
            return msg
        elif msg:
            return msg
        
    def _global_message(self, use_copies=None, msg=None):
        """Call stop_message on the internal nodes."""
        if use_copies is not None:
            self.use_copies = use_copies
        for (node, node_msg) in zip(self.nodes, self._get_split_msgs(msg)):
            node.global_message(node_msg)
     
    def bi_reset(self):
        """Call bi_reset on all the inner nodes."""
        if self.use_copies:
            for node in self.nodes:
                node.bi_reset()
        else:
            # note: reaching this code probably means that copies should be used
            self.node.bi_reset()
    
    def is_bi_training(self):
        """Return true if is_bi_learning is True for at least one inner node.
        
        But if copies are used it is still advisable that they return the same
        is_bi_learning value.
        """
        if self.use_copies:
            for node in self.nodes:
                if node.is_bi_training():
                    return True
            return False
        else:
            if self.node.is_bi_training():
                return True
            else:
                return False
    
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
    
    def _parse_msg_copy_flag(self, msg):
        """Look for the the possible copy flag and modify the msg if needed.
        
        If the copy flag is found the Node is switched accordingly.
        """
        msg, copy_kwarg, target = self._parse_message(msg, ["use_copies"])
        if (("use_copies" in copy_kwarg) and
            (copy_kwarg["use_copies"] is not None)):
            self.use_copies = copy_kwarg["use_copies"]
    
    def _get_split_msgs(self, msg):
        """Return messages for the individual nodes."""
        if not msg:
            return [None] * len(self.nodes)
        msgs = [dict() for _ in range(len(self.nodes))]
        for (key, value) in msg.items():
            if type(value) is n.ndarray:
                split_values = n.hsplit(value, len(self.nodes))
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
        msg = dict()
        for (key, first_value) in msgs[-1].items():
            if type(first_value) is n.ndarray:
                msg[key] = n.hstack([node_msg[key] for node_msg in msgs])
            else:
                # pick the msg value of the first node
                msg[key] = msgs[-1][key]
        return msg
