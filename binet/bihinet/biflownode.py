
import mdp
import mdp.hinet as hinet
n = mdp.numx

from ..binode import BiNode, BiNodeException
from ..biflow import BiFlow

# TODO: add derived BiFlowNode which allow specification message flag for
#    BiFlowNode to specify the internal target? Or hardwired target?


class BiFlowNode(BiNode, hinet.FlowNode):
    """BiFlowNode wraps a BiFlow of Nodes into a single BiNode.
    
    This is handy if you want to use a flow where a Node is required.
    Additional args and kwargs for train and execute are supported.
    
    Note that for nodes in the internal flow the intermediate training phases
    will generally be closed, e.g. a CheckpointSaveFunction should not expect
    these training  phases to be left open.
    
    All the read-only container slots are supported and are forwarded to the
    internal flow.
    """
    
    def __init__(self, biflow, input_dim=None, output_dim=None, dtype=None):
        """Wrap the given BiFlow into this node.
        
        Pretrained nodes are allowed, but the internal _flow should not 
        be modified after the BiFlowNode was created (this will cause problems
        if the training phase structure of the internal nodes changes).
        
        The node dimensions do not have to be specified. Unlike in a normal
        FlowNode they cannot be extracted from the nodes and are left unfixed.
        The data type is left unfixed as well.
        """
        if not isinstance(biflow, BiFlow):
            raise BiNodeException("The biflow has to be an BiFlow instance.")
        super(BiFlowNode, self).__init__(flow=biflow,
                                         input_dim=input_dim,
                                         output_dim=output_dim, dtype=dtype)
        # last successful request for target node_id, is used for reentry
        self._last_id_request = None
        
    def _get_target(self):
        """Return the last successfully requested target node_id.
        
        The stored target is then reset to None.
        If no target is stored (i.e. if it is None) then 0 is returned.
        """
        if self._last_id_request:
            target = self._last_id_request
            self._last_id_request = None
            return target
        else:
            return 0
        return 0
    
    def _target_for_reentry(self, target):
        """Try to translate the target value to a node_id and return it.
        
        If this is not possible because the target node has no node_id then
        an exception is raised.
        """
        # TODO: assign random target id to node if it has no node_id
        #    or do this even earlier? can turn it off via flag?
        if isinstance(target, int):
            if isinstance(self._flow[target], BiNode):
                str_target = self._flow[target].node_id
                if str_target is not None:
                    return str_target
            err = ("Flow reentry after global message is not possible since "
                   "the target value " + target + " refering to a node of "
                   "type " + self._flow[target] + " cannot be translated into "
                   "a node id.")
            raise BiNodeException(err)
        else:    
            return target
        
    def _get_method(self, method_name, default_method, target):
        """Return the default method and the target.
        
        This method overrides the standard BiNode _get_method to delegate the
        method selection to the internal nodes. If the method_name is
        'inverse' then adjustments are made so that the last internal node is
        called.
        """
        if method_name == "inverse": 
            if target == -1:
                target = None
            if target is None:
                self._last_id_request = len(self._flow) - 1
        return default_method, target
       
    def _execute(self, x, msg=None):
        target = self._get_target()
        result = self._flow._execute_seq(x, msg, target)
        if (isinstance(result, tuple) and (len(result) > 3) and
            isinstance(result[2], int)):
            target = self._target_for_reentry(result[2])
            result = result[:2] + (target,) + result[3:]
        return result
    
    def _get_execute_method(self, x, method_name, target):
        """Return _execute and the provided target.
        
        The method selection is done in the contained nodes.
        """
        return self._execute, target
    
    def _get_train_seq(self):
        """Return a training sequence containing all training phases."""
        train_seq = []
        for i_node, node in enumerate(self._flow):
            if node.is_trainable():
                remaining_len = (len(node._get_train_seq())
                                 - self._pretrained_phase[i_node])
                train_seq += ([(self._get_train_function(i_node), 
                                self._get_stop_training_function(i_node))] 
                              * remaining_len)
        # if the last node is trainable we have to set the output dimensions
        # to those of the BiFlowNode.
        if self._flow[-1].is_trainable():
            train_seq[-1] = (train_seq[-1][0],
                             self._get_stop_training_wrapper(self._flow[-1],
                                                             train_seq[-1][1]))
        return train_seq
    
    ## Helper methods for _get_train_seq. ##
    
    def _get_train_function(self, _i_node):
        """Internal function factory for train.
        
        _i_node -- the index of the node to be trained
        """
        # This method is similar to BiFlow._train_node_single_phase.
        def _train(x, msg=None):
            target = self._get_target()
            ## loop until we have to go outside or complete train
            while True:
                ## execute flow before training node
                result = self._flow._execute_seq(x, msg, target=target,
                                                 stop_at_node=_i_node)
                if isinstance(result, tuple):
                    if (len(result) == 2):
                        # did not reach the training node this time
                        return result
                    elif (len(result) == 3) and (result[2] is True):
                        # reached the training node
                        x = result[0]
                        msg = result[1]
                    else:
                        # message for outside, reenter flownode later
                        if isinstance(result[2], int):
                            target = self._target_for_reentry(result[2])
                            result = result[:2] + (target,) + result[3:]
                        return result
                else:
                    # did not reach the training node this time, return y
                    return result
                ## perform node training
                if isinstance(self._flow[_i_node], BiNode):
                    result = self._flow[_i_node].train(x, msg)
                else:
                    self._flow[_i_node].train(x)
                    result = None
                ## process the training result
                if not result:
                    return None
                elif isinstance(result, dict):
                    self._flow._global_message_seq(result, ignore_node=_i_node)
                    return result
                elif len(result) == 2:
                    branch_msg, branch_target = result
                    return self._flow._branch_message_seq(msg=branch_msg,
                                                          target=branch_target,
                                                          current_node=_i_node)
                elif len(result) == 3:
                    x, msg, target = result
                    continue
                # deal with combination of branch and continued execution
                if len(result) == 4:
                    branch_result = result[3]
                    self._flow._global_message_seq(branch_result,
                                                   ignore_node=_i_node)
                else:
                    branch_result = self._flow._branch_message_seq(
                                                    msg=result[3],
                                                    target=result[4],
                                                    current_node=_i_node)
                if branch_result:
                    # message for outside, store target for reenter
                    if isinstance(result[2], int):
                        target = self._target_for_reentry(result[2])
                        result = result[:2] + (target,) + result[3:]
                    return result
                else:
                    x, msg, target = result[:3]
                    continue
                err = ("Internal node produced invalid return "
                       "value for train: " + str(result))
                raise BiNodeException(err)
        # return the custom _train function
        return _train
    
    def _get_stop_training_function(self, _i_node):
        """Internal function factory for stop_training.
        
        _i_node -- the index of the node for which the training stops
        """
        def _stop_training(msg=None):
            # run stop_bi_training locally for relative target
            if msg:
                result = self._flow[_i_node].stop_training(msg)
            else:
                result = self._flow[_i_node].stop_training()
            # process stop_training result
            if not result:
                return None
            elif isinstance(result, dict):
                self._flow._global_message_seq(result, ignore_node=_i_node)
                return result
            elif len(result) == 2:
                return self._flow._stop_message_seq(msg=result[0],
                                                    target=result[1],
                                                    current_node=_i_node)
            err = ("Internal node produced invalid return "
                   "value for stop_training: " + str(result))
            raise BiNodeException(err)
        # return the custom _stop_training function
        return _stop_training
    
    def _get_stop_training_wrapper(self, node, func):
        """Return wrapper for stop_training to set BiFlowNode outputdim."""
        # We have to overwrite the version from FlowNode to take care of the
        # optional return value.
        def _stop_training_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            self.output_dim = node.output_dim
            return result
        return _stop_training_wrapper
    
    ### Special BiNode methods ###
    
    def _message(self, msg=None):
        target = self._get_target()
        return self._flow._branch_message_seq(msg, target)
    
    def _stop_message(self, msg=None):
        target = self._get_target()
        return self._flow._stop_message_seq(msg, target)
    
    def bi_reset(self):
        self._last_id_request = None
        for node in self._flow:
            if isinstance(node, BiNode):
                node.bi_reset()
        super(BiFlowNode, self).bi_reset()
    
    def is_bi_training(self):
        for node in self._flow:
            if isinstance(node, BiNode) and node.is_bi_training():
                return True
        return False    
    
    def _request_node_id(self, node_id):
        for node in self._flow:
            if isinstance(node, BiNode):
                found_node = node._request_node_id(node_id)
                if found_node:
                    self._last_id_request = node_id
                    return found_node
        return None
            
        
