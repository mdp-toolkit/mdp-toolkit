from builtins import str

import mdp
import mdp.hinet as hinet
n = mdp.numx

from bimdp import BiNode, BiNodeException
from bimdp import BiFlow, BiFlowException

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

    def __init__(self, biflow, input_dim=None, output_dim=None, dtype=None,
                 node_id=None):
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
                                         output_dim=output_dim, dtype=dtype,
                                         node_id=node_id)
        # last successful request for target node_id
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

    def _get_method(self, method_name, default_method, target):
        """Return the default method and the target.

        This method overrides the standard BiNode _get_method to delegate the
        method selection to the internal nodes. If the method_name is
        'inverse' then adjustments are made so that the last internal node is
        called.
        """
        if method_name == "inverse":
            if self._last_id_request is None:
                if target == -1:
                    target = None
                if target is None:
                    self._last_id_request = len(self._flow) - 1
        return default_method, target

    def _execute(self, x, msg=None):
        target = self._get_target()
        i_node = self._flow._target_to_index(target)
        # we know that _get_target returned a valid target, so no check
        return self._flow._execute_seq(x, msg, i_node)

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

    def _get_train_function(self, nodenr):
        """Internal function factory for train.

        nodenr -- the index of the node to be trained
        """
        # This method is similar to the first part of
        # BiFlow._train_node_single_phase.
        def _train(x, msg=None):
            target = self._get_target()
            i_node = self._flow._target_to_index(target)
            ## loop until we have to go outside or complete train
            while True:
                ## execute flow before training node
                result = self._flow._execute_seq(x, msg, i_node=i_node,
                                                 stop_at_node=nodenr)
                if (isinstance(result, tuple) and len(result) == 3 and
                    result[2] is True):
                    # we have reached the training node
                    x = result[0]
                    msg = result[1]
                    i_node = nodenr  # have to update this manually
                else:
                    # flownode should be reentered later
                    return result
                ## perform node training
                if isinstance(self._flow[nodenr], BiNode):
                    result = self._flow[nodenr].train(x, msg)
                    if result is None:
                        return None  # training is done for this chunk
                else:
                    self._flow[nodenr].train(x)
                    return None
                ## training execution continues, interpret result
                if not isinstance(result, tuple):
                    x = result
                    msg = None
                    target = None
                elif len(result) == 2:
                    x, msg = result
                    target = None
                elif len(result) == 3:
                    x, msg, target = result
                else:
                    # reaching this is probably an error, leave the handling
                    # to the outer flow
                    return result
                ## check if the target is in this flow, return otherwise
                if isinstance(target, int):
                    i_node = i_node + target
                    # values of +1 and -1 beyond this flow are allowed
                    if i_node == len(self._flow):
                        if not msg:
                            return x
                        else:
                            return (x, msg)
                    elif i_node == -1:
                        return x, msg, -1
                else:
                    i_node = self._flow._target_to_index(target, i_node)
                    if not isinstance(i_node, int):
                        # target not found in this flow
                        # this is also the exit point when EXIT_TARGET is given
                        return x, msg, target
        # return the custom _train function
        return _train

    def _get_stop_training_function(self, nodenr):
        """Internal function factory for stop_training.

        nodenr -- the index of the node for which the training stops
        """
        # This method is similar to the second part of
        # BiFlow._train_node_single_phase.
        def _stop_training(msg=None):
            if isinstance(self._flow[nodenr], BiNode):
                result = self._flow[nodenr].stop_training(msg)
            else:
                # for a non-bi Node the msg is dropped
                result = self._flow[nodenr].stop_training()
            # process stop_training result
            if result is None:
                return None
            # prepare execution phase
            if not isinstance(result, tuple):
                x = result
                msg = None
                target = None
            elif len(result) == 2:
                x, msg = result
                target = None
            elif len(result) == 3:
                x, msg, target = result
            else:
                err = ("Node produced invalid return value " +
                       "for stop_training: " + str(result))
                raise BiFlowException(err)
            if isinstance(target, int):
                i_node = nodenr + target
                # values of +1 and -1 beyond this flow are allowed
                if i_node == len(self._flow):
                    return x, msg, 1
                elif i_node == -1:
                    return x, msg, -1
            else:
                i_node = self._flow._target_to_index(target, nodenr)
                if not isinstance(i_node, int):
                    # target not found in this flow
                    # this is also the exit point when EXIT_TARGET is given
                    return x, msg, target
            return self._flow._execute_seq(x, msg, i_node=i_node)
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

    def _bi_reset(self):
        self._last_id_request = None
        for node in self._flow:
            if isinstance(node, BiNode):
                node.bi_reset()

    def _request_node_id(self, node_id):
        if self._node_id == node_id:
            return self
        for node in self._flow:
            if isinstance(node, BiNode):
                found_node = node._request_node_id(node_id)
                if found_node:
                    self._last_id_request = node_id
                    return found_node
        return None
